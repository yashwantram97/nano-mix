"""Check that download.py finished cleanly and fully. Reads data/raw/ only; never changes it.

    uv run nanomix-verify-download [--offline] [--deep] [--workers N]

1. Complete: every non-eval file of each subset is on disk at its exact size (compared with a fresh
   HF listing; with --offline, with manifest.json), and manifest.json says "full".
2. Clean: no eval-only shard, no leftover .partial/ or .tmp, no unexpected files.
3. Readable: every parquet footer opens. With --deep, every file is read through to the end
   (all parquet row groups, every .json.gz line parsed), which catches corruption inside a file.
4. python_edu content: every planned content file exists; it holds only its planned blob_ids, in
   order, each once across all files; every missing blob_id is listed in failed_blob_ids.txt; no
   holdout blob_id; failures under MAX_FAILED_SHARE.
"""

import argparse
import gzip
import json
import os
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from nanomix.data.download import list_files, python_edu_chunks, python_edu_holdout
from nanomix.paths import RAW
from nanomix.sources import EVAL_ONLY, PILOT_DUPLICATE, PYTHON_EDU_EVAL_BLOBS, SOURCES

MAX_FAILED_SHARE = 0.001  # blobs S3 couldn't return; the pilot lost 1 in 250,000
CONTENT_COLUMNS = ["blob_id", "repo_name", "path", "score", "text"]


def dest_dir(name):
    return RAW / name / "metadata" if name == "python_edu" else RAW / name


def read_fully(path):
    """Read a whole file; returns (path, documents, error or None)."""
    try:
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            rows = sum(pf.read_row_group(i).num_rows for i in range(pf.num_row_groups))
            return str(path), rows, None if rows == pf.metadata.num_rows else "row count != footer"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = 0
            for line in f:
                if "text" not in json.loads(line):
                    return str(path), rows, f"line {rows + 1} has no text"
                rows += 1
        return str(path), rows, None
    except Exception as e:
        return str(path), 0, f"{type(e).__name__}: {e}"


def check_hf_files(name, manifest, offline):
    """Checks 1-3 for one source's HF files. Returns (problems, local paths)."""
    problems = []
    entry = manifest.get(name)
    if entry is None:
        return [f"{name}: not in manifest.json"], []
    if entry.get("status") != "full":
        problems.append(f"{name}: manifest status is {entry.get('status')!r}, not 'full'")

    expected = [(f["path"], f["bytes"]) for f in entry["files"]]
    if not offline:
        try:
            listed = list_files(name)  # also refuses an eval-only shard on disk
        except SystemExit as e:
            return [f"{name}: {e}"], []
        if sorted(listed) != sorted(expected):
            problems.append(f"{name}: manifest.json files differ from the HF listing "
                            f"({len(expected)} vs {len(listed)} files)")
        expected = listed

    eval_only = EVAL_ONLY.get(name)
    if eval_only in {Path(p).name for p, _ in expected}:
        problems.append(f"{name}: eval-only shard {eval_only} is in the file list")
    local = []
    for path, size in expected:
        f = dest_dir(name) / Path(path).name
        if not f.exists():
            problems.append(f"{name}: missing {f}")
        elif f.stat().st_size != size:
            problems.append(f"{name}: {f} is {f.stat().st_size:,} bytes, expected {size:,}")
        else:
            local.append(f)
    return problems, local


def check_clean(name, allowed):
    """Check 2: nothing in data/raw/<name>/ except the allowed files."""
    problems = []
    for f in sorted((RAW / name).rglob("*")):
        if f.name == ".partial" or f.suffix == ".tmp" or ".partial" in f.parts:
            problems.append(f"{name}: leftover from an interrupted run: {f}")
        elif f.name == EVAL_ONLY.get(name):
            problems.append(f"{name}: eval-only shard on disk: {f}")
        elif f.is_file() and f not in allowed:
            problems.append(f"{name}: unexpected file {f}")
    return problems


def check_python_edu_content(metadata_files):
    """Check 4. Returns (problems, notes, content paths)."""
    problems, notes = [], []
    content = RAW / "python_edu" / "content"
    failed_log = RAW / "python_edu" / "failed_blob_ids.txt"
    failed = set(failed_log.read_text().split()) if failed_log.exists() else set()
    holdout = python_edu_holdout(metadata_files)
    blob_ids = {p: pq.read_table(p, columns=["blob_id"]).column("blob_id").to_pylist() for p in metadata_files}

    seen, planned_total, fetched_total, missing_total, paths = set(), 0, 0, 0, []
    for name, metadata_file, rows in tqdm(python_edu_chunks(metadata_files), desc="  content files", unit="file"):
        f = content / name
        if not f.exists():
            problems.append(f"python_edu: missing content file {f}")
            continue
        paths.append(f)
        schema = pq.ParquetFile(f).schema_arrow.names
        if schema != CONTENT_COLUMNS:
            problems.append(f"python_edu: {f.name} has columns {schema}, expected {CONTENT_COLUMNS}")
            continue
        planned = [blob_ids[metadata_file][i] for i in rows]
        got = pq.read_table(f, columns=["blob_id"]).column("blob_id").to_pylist()
        got_set = set(got)
        # The file must be the planned list with only the failed blobs taken out, in the same order.
        if [b for b in planned if b in got_set] != got:
            problems.append(f"python_edu: {f.name} holds blob_ids that aren't its planned rows, or out of order")
        missing = [b for b in planned if b not in got_set]
        unexplained = [b for b in missing if b not in failed]
        if unexplained:
            problems.append(f"python_edu: {f.name} is missing {len(unexplained):,} blobs not in {failed_log.name}")
        dup = got_set & seen
        if dup:
            problems.append(f"python_edu: {f.name} repeats {len(dup):,} blob_ids from earlier files")
        if got_set & holdout:
            problems.append(f"python_edu: {f.name} contains {len(got_set & holdout):,} holdout blob_ids")
        seen |= got_set
        planned_total += len(planned)
        fetched_total += len(got)
        missing_total += len(missing)

    share = missing_total / planned_total if planned_total else 0
    notes.append(f"python_edu: {fetched_total:,} of {planned_total:,} planned blobs fetched, "
                 f"{missing_total:,} failed ({share:.4%})")
    if share > MAX_FAILED_SHARE:
        problems.append(f"python_edu: {share:.3%} of blobs failed (> {MAX_FAILED_SHARE:.1%}); re-fetch them")
    if (content / PILOT_DUPLICATE).exists():
        notes.append(f"python_edu: {PILOT_DUPLICATE} is a duplicate of the first 5,000 rows of the first "
                     "content file; tokenize_data.py must skip it")
    return problems, notes, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="compare with manifest.json instead of a fresh HF listing")
    parser.add_argument("--deep", action="store_true", help="read every file through to the end")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes for --deep")
    args = parser.parse_args()

    manifest_path = RAW / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found; run download.py first")
    manifest = json.loads(manifest_path.read_text())

    problems, notes, results, to_read = [], [], {}, []
    for name in SOURCES:
        print(f"checking {name}...")
        p, local = check_hf_files(name, manifest, args.offline)
        allowed = set(local)
        if name == "python_edu" and local:
            if not PYTHON_EDU_EVAL_BLOBS.exists():
                notes.append(f"python_edu: {PYTHON_EDU_EVAL_BLOBS} not here; holdout checked against the reserved rows only")
            p4, n4, content_files = check_python_edu_content(local)
            p += p4
            notes += n4
            allowed |= set(content_files) | {RAW / name / "failed_blob_ids.txt", RAW / name / "content" / PILOT_DUPLICATE}
            local += content_files
        p += check_clean(name, allowed)

        # 3. Readable: footers always; whole files with --deep.
        for f in local:
            if f.suffix == ".parquet":
                try:
                    pq.ParquetFile(f).metadata
                except Exception as e:
                    p.append(f"{name}: can't open {f}: {e}")
        to_read += local
        results[name] = (p, len(local), sum(f.stat().st_size for f in local))
        problems += p

    if args.deep:
        print(f"\nreading all {len(to_read)} files with {args.workers} processes...")
        # Largest first, so the long gutenberg/fineweb files don't all start at the end.
        to_read.sort(key=lambda f: f.stat().st_size, reverse=True)
        sizes = {str(f): f.stat().st_size for f in to_read}
        with Pool(args.workers) as pool, tqdm(total=sum(sizes.values()), unit="B", unit_scale=True, desc="deep read") as bar:
            for path, rows, error in pool.imap_unordered(read_fully, to_read):
                bar.update(sizes[path])
                if error:
                    name = Path(path).relative_to(RAW).parts[0]
                    results[name][0].append(f"{name}: {path}: {error}")
                    problems.append(f"{name}: {path}: {error}")

    print(f"\n{'source':12s} {'files':>5s} {'GB':>7s}  result")
    for name, (p, n, size) in results.items():
        print(f"{name:12s} {n:5d} {size / 1e9:7.2f}  {'PASS' if not p else f'FAIL ({len(p)})'}")
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print("\nProblems:")
        for p in problems[:50]:
            print(f"  {p}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        raise SystemExit("\nFAILED")
    print(f"\nAll checks passed{' (deep read)' if args.deep else ''}.")


if __name__ == "__main__":
    main()
