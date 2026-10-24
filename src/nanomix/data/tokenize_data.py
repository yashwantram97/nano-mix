"""Tokenize every raw file of each source into flat uint16 token files in data/tokenized/<source>/
for OLMo-core, one .npy per raw file with the same base name.

Each document -> tokens (no special tokens) + EOS. No BOS, no padding, no chunking.
Files whose .npy already exists are skipped (and still listed in meta.json), so an interrupted
run can be re-run.

    uv run nanomix-tokenize --source {fineweb_edu,finemath,python_edu,gutenberg,all} [--workers N]
"""

import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from nanomix.paths import RAW, TOKENIZED as OUT
from nanomix.sources import DTYPE, EVAL_ONLY, PILOT_DUPLICATE, SOURCES, TOKENIZER

# Parallelism comes from one process per input file; keep each tokenizer single-threaded.
# (transformers is imported later, inside get_tokenizer(), so setting this here is early enough.)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GB = 1e9


def input_files(source):
    """All raw files of a source, sorted. Refuses eval-only shards; skips the duplicate pilot file."""
    files = sorted(p for p in RAW.glob(SOURCES[source].raw_glob) if p.name != PILOT_DUPLICATE)
    eval_only = [p for p in files if p.name == EVAL_ONLY.get(source)]
    if eval_only:
        raise SystemExit(f"{eval_only[0]} is reserved for the holdout set and must not be tokenized")
    return files


def read_batches(path, batch_size):
    """Yield (document texts, raw bytes read so far) from a parquet or .json.gz file."""
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        size, rows, done = path.stat().st_size, pf.metadata.num_rows, 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
            done += batch.num_rows
            yield batch.column("text").to_pylist(), size * done // max(rows, 1)
    else:  # .json.gz, one JSON document per line; progress is the position in the compressed file
        with open(path, "rb") as raw, io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8") as lines:
            texts = []
            for line in lines:
                texts.append(json.loads(line)["text"])
                if len(texts) == batch_size:
                    yield texts, raw.tell()
                    texts = []
            if texts:
                yield texts, raw.tell()


_tokenizer = None
_progress = None  # in each worker process: shared count of raw bytes done, read by the main process


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()  # silences the "sequence length is longer than..." warning
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tokenizer


def _init_worker(counter):
    global _progress
    _progress = counter


def _report(nbytes):
    with _progress.get_lock():
        _progress.value += nbytes


def tokenize_file(job):
    """Tokenize one input file into out_dir/<name>.npy. Returns (source, stats for meta.json)."""
    source, path, out_dir, batch_size = job
    out = out_dir / (path.name.split(".")[0] + ".npy")
    size = path.stat().st_size
    if out.exists():
        tokens = np.memmap(out, dtype=DTYPE, mode="r")
        _report(size)
        return source, {"file": out.name, "raw": str(path), "num_tokens": len(tokens),
                        "num_docs": int((tokens == get_tokenizer().eos_token_id).sum()), "skipped": True}

    tok = get_tokenizer()
    eos = tok.eos_token_id
    num_tokens = num_docs = reported = 0
    t0 = time.time()
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        for texts, done in read_batches(path, batch_size):
            texts = [t for t in texts if t]  # skip empty / None
            # split_special_tokens: a literal "<|endoftext|>" inside a document must not become EOS.
            ids = tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            arr = np.fromiter((i for doc in ids for i in (*doc, eos)), dtype=np.int64)
            assert arr.max(initial=0) < 2**16, "token id does not fit in uint16"
            arr.astype(DTYPE).tofile(f)  # raw values, no .npy header
            num_tokens += len(arr)
            num_docs += len(ids)
            _report(min(done, size) - reported)
            reported = min(done, size)
    _report(size - reported)
    os.replace(tmp, out)
    return source, {"file": out.name, "raw": str(path), "num_tokens": num_tokens, "num_docs": num_docs,
                    "seconds": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[*SOURCES, "all"])
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()
    names = list(SOURCES) if args.source == "all" else [args.source]

    jobs = []
    for name in names:
        files = input_files(name)
        if not files:
            print(f"{name}: no input files in {RAW / SOURCES[name].raw_glob}, skipping")
        (OUT / name).mkdir(parents=True, exist_ok=True)
        jobs += [(name, path, OUT / name, SOURCES[name].batch_docs) for path in files]
        # mix.py globs every .npy in the folder, so a token file with no raw file would end up in the mix.
        expected = {p.name.split(".")[0] + ".npy" for p in files}
        stale = sorted(p.name for p in (OUT / name).glob("*.npy") if p.name not in expected)
        if stale:
            print(f"{name}: WARNING, token files with no raw file (remove them before mix.py): {stale}")

    # Largest files first, so the big ones don't end up running alone at the end.
    jobs.sort(key=lambda job: job[1].stat().st_size, reverse=True)
    sizes = sum(job[1].stat().st_size for job in jobs)
    done_files = sum((job[2] / (job[1].name.split(".")[0] + ".npy")).exists() for job in jobs)
    print(f"Tokenizing {len(jobs)} files ({sizes / GB:.1f} GB raw, {done_files} already done) "
          f"with {args.workers} workers -> {OUT}/")

    t0 = time.time()
    results = {name: [] for name in names}
    counter = mp.Value("q", 0)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(counter,)) as pool, \
            tqdm(total=sizes, unit="B", unit_scale=True, desc="tokenize", smoothing=0.05) as bar:
        pending = [pool.apply_async(tokenize_file, (job,)) for job in jobs]
        reported, files_done = 0, 0
        while pending:
            time.sleep(1)
            done = counter.value
            bar.update(done - reported)
            reported = done
            for result in [r for r in pending if r.ready()]:
                pending.remove(result)
                name, stats = result.get()  # re-raises a worker's error
                results[name].append(stats)
                files_done += 1
            bar.set_postfix_str(f"files {files_done}/{len(jobs)}")
    wall = time.time() - t0

    manifest = json.loads((RAW / "manifest.json").read_text())
    eos = get_tokenizer().eos_token_id
    print(f"\n{'source':12s} {'files':>5s} {'tokens':>16s} {'docs':>12s} {'GB':>7s}")
    new_tokens = 0
    for name in names:
        files = sorted(results[name], key=lambda f: f["file"])
        if not files:
            continue
        meta = {
            "repo": manifest[name]["repo"],
            "commit": manifest[name]["revision"],
            "tokenizer": TOKENIZER,
            "dtype": "uint16",
            "eos_id": eos,
            "num_tokens": sum(f["num_tokens"] for f in files),
            "num_docs": sum(f["num_docs"] for f in files),
            "files": [{k: f[k] for k in ("file", "raw", "num_tokens", "num_docs")} for f in files],
        }
        (OUT / name / "meta.json").write_text(json.dumps(meta, indent=2))
        new_tokens += sum(f["num_tokens"] for f in files if not f.get("skipped"))
        print(f"{name:12s} {len(files):5d} {meta['num_tokens']:16,} {meta['num_docs']:12,} {meta['num_tokens'] * 2 / GB:7.2f}")

    print(f"\nwall time {wall / 60:.1f} min; {new_tokens:,} new tokens ({new_tokens / max(wall, 1):,.0f} tokens/s)")


if __name__ == "__main__":
    main()
