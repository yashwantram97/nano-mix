"""Full download: every file of each source's subset into data/raw/, except the eval-only shards.

Raw data only, no processing. Files already on disk are skipped, so an interrupted run can be re-run.

    uv run python download.py --source {fineweb_edu,finemath,gutenberg,python_edu,all} [--workers N]

python_edu's metadata comes from HF; its code is fetched from Software Heritage S3 in chunks of
CHUNK_ROWS metadata rows, one parquet file per chunk. Needs data/raw_eval/python_edu.parquet.
"""

import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

RAW = Path("data/raw")
GB = 1e9

# Each repo is pinned to a commit so the download is reproducible.
SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "data/CC-MAIN-2025-26/*.parquet"),
    "finemath": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594", "finemath-4plus/*.parquet"),
    "gutenberg": ("common-pile/project_gutenberg_filtered", "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7", "*.json.gz"),
    "python_edu": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", "python-edu/*.parquet"),
}

# Shards reserved for the frozen holdout sets (CLAUDE.md). Never downloaded for training.
EVAL_ONLY = {
    "fineweb_edu": "000_00049.parquet",
    "finemath": "train-00063-of-00064.parquet",
    "gutenberg": "project_gutenberg-dolma-0014.json.gz",
}

# python_edu's holdout is the last 8,000 rows of this metadata file; the blob_ids it used are in
# data/raw_eval/python_edu.parquet. Those rows are never fetched, and those blob_ids are skipped anywhere.
PYTHON_EDU_EVAL_FILE, PYTHON_EDU_EVAL_ROWS = "train-00000-of-00002.parquet", 8_000
PYTHON_EDU_EVAL_BLOBS = Path("data/raw_eval/python_edu.parquet")
CHUNK_ROWS = 250_000  # metadata rows per content file, ~190 MB and ~127M tokens each
PYTHON_EDU_ROWS = 7_678_448  # rows in both metadata files
CONTENT_BYTES_PER_ROW = 770  # from the pilot (192 MB for 250k rows), for the disk check and progress
PILOT_CONTENT = "pilot-250000.parquet"  # the pilot's rows 0-250k of PYTHON_EDU_EVAL_FILE, reused as that chunk
PILOT_DUPLICATE = "pilot-5000.parquet"  # an earlier pilot: the first 5,000 rows of PILOT_CONTENT again


def list_files(name):
    """All non-eval repo files matching the source's pattern, as (path, size), sorted by path."""
    repo, revision, pattern = SOURCES[name]
    tree = HfApi().list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
    files = sorted((f.path, f.size) for f in tree if hasattr(f, "size") and fnmatch(f.path, pattern))
    eval_only = EVAL_ONLY.get(name)
    if eval_only:
        if eval_only not in {Path(p).name for p, _ in files}:
            raise SystemExit(f"{name}: eval-only shard {eval_only} not in the listing; check SOURCES / EVAL_ONLY")
        if (RAW / name / eval_only).exists():
            raise SystemExit(f"{RAW / name / eval_only} is eval-only and must not be in data/raw/; remove it")
        files = [(p, s) for p, s in files if Path(p).name != eval_only]
    return files


def download_file(repo, revision, repo_path, size, dest_dir):
    """Download one file to dest_dir. Skips it if already complete. Returns seconds spent (0 if skipped)."""
    final = dest_dir / Path(repo_path).name
    if final.exists() and final.stat().st_size == size:
        return 0.0

    # Download into .partial/ first and only move into place once the size is right,
    # so a half-finished file is never mistaken for a complete one.
    t0 = time.time()
    tmp = Path(hf_hub_download(repo, repo_path, repo_type="dataset", revision=revision, local_dir=dest_dir / ".partial"))
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{repo_path}: size {tmp.stat().st_size} != expected {size}")
    os.replace(tmp, final)
    shutil.rmtree(dest_dir / ".partial")
    return time.time() - t0


def count_docs(path):
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    with gzip.open(path) as f:  # .json.gz: one document per line
        return sum(1 for _ in f)


# ---------------------------------------------------------------- python_edu content from S3


def s3_client(workers, test_blob_id):
    """Try anonymous access first, then the AWS credentials from the standard boto3 config."""
    pool = Config(max_pool_connections=workers, retries={"max_attempts": 5, "mode": "adaptive"})
    for mode, config in [("anonymous", pool.merge(Config(signature_version=UNSIGNED))), ("aws-credentials", pool)]:
        client = boto3.client("s3", region_name="us-east-1", config=config)
        try:
            client.get_object(Bucket="softwareheritage", Key=f"content/{test_blob_id}")
            return client, mode
        except Exception as e:
            print(f"  S3 {mode} access failed: {e}")
    return None, None


def fetch_blob(client, blob_id):
    """Return the file's text, or None if it can't be fetched. botocore retries with backoff."""
    try:
        body = client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")["Body"].read()
        return gzip.decompress(body).decode("utf-8", errors="ignore")  # as in the dataset card
    except Exception:
        return None


def python_edu_chunks(metadata_files):
    """Plan the content files: (output name, metadata file, rows) per chunk of CHUNK_ROWS rows.

    rows are the metadata rows to fetch: not in the holdout, and the first occurrence of each
    blob_id across both metadata files (130 blob_ids repeat).
    """
    if not PYTHON_EDU_EVAL_BLOBS.exists():
        raise SystemExit(f"{PYTHON_EDU_EVAL_BLOBS} is needed to exclude the holdout blob_ids; copy it here first")
    seen = set(pq.read_table(PYTHON_EDU_EVAL_BLOBS, columns=["ref"]).column("ref").to_pylist())
    chunks = []
    for path in metadata_files:
        blob_ids = pq.read_table(path, columns=["blob_id"]).column("blob_id").to_pylist()
        end = len(blob_ids) - PYTHON_EDU_EVAL_ROWS if path.name == PYTHON_EDU_EVAL_FILE else len(blob_ids)
        for start in range(0, end, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, end)
            rows = []
            for i in range(start, stop):
                if blob_ids[i] not in seen:
                    seen.add(blob_ids[i])
                    rows.append(i)
            name = f"{path.stem}_rows-{start:07d}-{stop:07d}.parquet"
            if path.name == PYTHON_EDU_EVAL_FILE and start == 0:
                name = PILOT_CONTENT
            chunks.append((name, path, rows))
    return chunks


def content_rows_on_disk(out_dir):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in (out_dir / "content").glob("*.parquet")
               if p.name != PILOT_DUPLICATE)


def fetch_chunk(client, metadata_file, rows, out, failed_log, workers, bars):
    """Fetch one chunk's rows from S3 into the parquet file out. Returns (fetched, failed)."""
    meta = pq.read_table(metadata_file, columns=["blob_id", "repo_name", "path", "score"]).take(rows).to_pylist()
    texts = []
    with ThreadPoolExecutor(workers) as pool:
        for text in pool.map(lambda row: fetch_blob(client, row["blob_id"]), meta):
            texts.append(text)
            bars["python_edu"].update(1)
            bars["overall"].update(CONTENT_BYTES_PER_ROW)

    ok = [(row, text) for row, text in zip(meta, texts) if text is not None]
    table = pa.table({
        "blob_id": [row["blob_id"] for row, _ in ok],
        "repo_name": [row["repo_name"] for row, _ in ok],
        "path": [row["path"] for row, _ in ok],
        "score": [row["score"] for row, _ in ok],
        "text": [text for _, text in ok],
    })
    pq.write_table(table, out.with_suffix(".tmp"))
    os.replace(out.with_suffix(".tmp"), out)
    failed = [row["blob_id"] for row, text in zip(meta, texts) if text is None]
    with open(failed_log, "a") as f:
        f.write("".join(f"{b}\n" for b in failed))
    return len(ok), len(failed)


def fetch_python_edu(chunks, out_dir, workers, overall, estimated_rows):
    """Fetch every chunk not yet on disk. Returns (manifest entries per chunk, total rows, seconds), or None without S3 access.

    estimated_rows is what the overall bar assumed before the chunks were planned; it is corrected here.
    """
    content = out_dir / "content"
    content.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chunks if not (content / c[0]).exists()]
    todo_rows = sum(len(rows) for _, _, rows in todo)
    overall.total += (todo_rows - estimated_rows) * CONTENT_BYTES_PER_ROW
    overall.refresh()
    tqdm.write(f"  {len(chunks)} content files, {len(chunks) - len(todo)} already downloaded, {todo_rows:,} rows to fetch")

    t0 = time.time()
    if todo:
        first_meta, first_rows = todo[0][1], todo[0][2]
        test_blob = pq.read_table(first_meta, columns=["blob_id"]).column("blob_id")[first_rows[0]].as_py()
        client, mode = s3_client(workers, test_blob)
        if client is None:
            tqdm.write(
                "  No S3 access. To fix: run `aws configure` with any AWS account's keys (region us-east-1),\n"
                "  check with `aws s3 ls s3://softwareheritage/content/ --max-items 1`, then re-run. Skipping python_edu content."
            )
            return None
        tqdm.write(f"  S3 access: {mode}")
        planned = sum(len(rows) for _, _, rows in chunks)
        bars = {"overall": overall,
                "python_edu": tqdm(total=planned, initial=planned - todo_rows, unit="file", desc="python_edu S3")}
        for i, (name, metadata_file, rows) in enumerate(todo, 1):
            bars["python_edu"].set_postfix_str(f"content file {i}/{len(todo)}")
            fetched, failed = fetch_chunk(client, metadata_file, rows, content / name, out_dir / "failed_blob_ids.txt", workers, bars)
            tqdm.write(f"  {name}: fetched {fetched:,}, failed {failed:,}")
        bars["python_edu"].close()

    entries = [{"file": name, "metadata": path.name, "rows_planned": len(rows),
                "rows_fetched": pq.ParquetFile(content / name).metadata.num_rows} for name, path, rows in chunks]
    return entries, sum(e["rows_fetched"] for e in entries), time.time() - t0


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[*SOURCES, "all"])
    parser.add_argument("--workers", type=int, default=32, help="parallel S3 fetches for python_edu")
    args = parser.parse_args()
    names = list(SOURCES) if args.source == "all" else [args.source]
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. List each repo and check there is enough disk for what is still missing (+20% headroom).
    listings = {name: list_files(name) for name in names}
    missing, need, content_rows_left = {}, 0, 0
    print("To download:")
    for name in names:
        dest_dir = RAW / name / "metadata" if name == "python_edu" else RAW / name
        missing[name] = {p for p, _ in listings[name] if not (dest_dir / Path(p).name).exists()}
        missing_bytes = sum(s for p, s in listings[name] if p in missing[name])
        if name == "python_edu":  # the S3 content isn't listed; estimate it from the rows not fetched yet
            content_rows_left = max(0, PYTHON_EDU_ROWS - PYTHON_EDU_EVAL_ROWS - content_rows_on_disk(RAW / name))
            missing_bytes += content_rows_left * CONTENT_BYTES_PER_ROW
        need += missing_bytes
        print(f"  {name:12s} {len(missing[name]):3d} of {len(listings[name]):3d} files missing, ~{missing_bytes / GB:6.2f} GB")
    free = shutil.disk_usage(RAW).free
    print(f"  with 20% headroom: {need * 1.2 / GB:.2f} GB, free: {free / GB:.2f} GB")
    if free < need * 1.2:
        raise SystemExit("Not enough disk space.")

    # 2. Download every listed file of each source.
    # The overall bar counts bytes still to download across all sources (python_edu's S3 content as
    # CONTENT_BYTES_PER_ROW per file), so its time left covers the whole run. HF files count when they
    # finish; huggingface_hub shows its own bar for the file in progress.
    overall = tqdm(total=need, unit="B", unit_scale=True, desc="overall", smoothing=0.05)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = []
    for name in names:
        repo, revision, _ = SOURCES[name]
        tqdm.write(f"\n{name}: {repo} @ {revision[:8]}  {len(listings[name])} files, {len(missing[name])} to download")
        out_dir = RAW / name
        dest_dir = out_dir / "metadata" if name == "python_edu" else out_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        total = sum(s for _, s in listings[name])
        done = sum(s for p, s in listings[name] if p not in missing[name])
        with tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=name) as bar:
            for i, (path, size) in enumerate(listings[name], 1):
                if path not in missing[name]:
                    continue
                bar.set_postfix_str(f"file {i}/{len(listings[name])}")
                download_file(repo, revision, path, size, dest_dir)
                bar.update(size)
                overall.update(size)
        local_files = [dest_dir / Path(p).name for p, _ in listings[name]]
        status = "full"
        entry = {
            "repo": repo,
            "revision": revision,
            "files": [{"path": p, "bytes": s} for p, s in listings[name]],
            "bytes": sum(s for _, s in listings[name]),
            "eval_only_excluded": EVAL_ONLY.get(name),
        }

        if name == "python_edu":
            docs = 0
            tqdm.write("  planning content files from the metadata...")
            result = fetch_python_edu(python_edu_chunks(local_files), out_dir, args.workers, overall, content_rows_left)
            if result is None:
                status = "metadata only (no S3 access)"
            else:
                entry["content"], docs, _ = result
                entry["eval_only_excluded"] = (f"last {PYTHON_EDU_EVAL_ROWS:,} rows of {PYTHON_EDU_EVAL_FILE} "
                                               f"and the blob_ids in {PYTHON_EDU_EVAL_BLOBS}")
        else:
            tqdm.write(f"  counting documents in {len(local_files)} files...")
            docs = sum(count_docs(f) for f in local_files)
        seconds = time.time() - t0

        manifest[name] = {**entry, "download_date": dt.date.today().isoformat(),
                          "download_seconds": round(seconds, 1), "status": status}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        disk = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        files_on_disk = sum(1 for f in out_dir.rglob("*") if f.is_file() and f.suffix != ".txt")
        summary.append((name, files_on_disk, disk, docs, seconds, status))
    overall.close()

    # 3. Report.
    print(f"\n{'source':12s} {'files':>5s} {'GB':>7s} {'documents':>11s} {'minutes':>8s}  status")
    for name, n, disk, docs, seconds, status in summary:
        print(f"{name:12s} {n:5d} {disk / GB:7.2f} {docs:11,} {seconds / 60:8.1f}  {status}")


if __name__ == "__main__":
    main()
