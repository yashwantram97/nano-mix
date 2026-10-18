"""Pilot tokenization: the whole first raw file of each source into flat uint16 token files
in data/tokenized/<source>/ for OLMo-core. Prints tokens/sec and full-run estimates.

Each document -> tokens (no special tokens) + EOS. No BOS, no padding, no chunking.

    uv run python tokenize_data.py --source {fineweb_edu,finemath,python_edu,gutenberg,all} [--workers N]
"""

import argparse
import gzip
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Parallelism comes from one process per input file; keep each tokenizer single-threaded.
# (transformers is imported later, inside get_tokenizer(), so setting this here is early enough.)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
DTYPE = np.uint16
RAW = Path("data/raw")
OUT = Path("data/tokenized")
GB = 1e9

# Input files per source, and how many documents to tokenize per batch (Gutenberg books are huge).
SOURCES = {
    "fineweb_edu": ("fineweb_edu/**/*.parquet", 1000),
    "finemath": ("finemath/**/*.parquet", 1000),
    "python_edu": ("python_edu/content/*.parquet", 1000),
    "gutenberg": ("gutenberg/**/*.json.gz", 8),
}

# Full-run sizes for the pilot estimate (from the HF listings used by download.py).
FULL_RAW_BYTES = {"fineweb_edu": 44.76e9, "finemath": 18.37e9, "gutenberg": 7.63e9}
PYTHON_EDU_DOCS = 7_678_448  # its raw content is fetched from S3, so estimate by document count


def input_files(source):
    return sorted(RAW.glob(SOURCES[source][0]))


def read_batches(path, batch_size):
    """Yield lists of document texts from a parquet or .json.gz file."""
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
            yield batch.column("text").to_pylist()
    else:  # .json.gz, one JSON document per line
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            texts = []
            for line in lines:
                texts.append(json.loads(line)["text"])
                if len(texts) == batch_size:
                    yield texts
                    texts = []
            if texts:
                yield texts


_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()  # silences the "sequence length is longer than..." warning
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tokenizer


def tokenize_file(job):
    """Tokenize one input file into out_dir/<name>.npy. Returns stats for meta.json."""
    path, out_dir, batch_size = job
    out = out_dir / (path.name.split(".")[0] + ".npy")
    if out.exists():
        tokens = np.memmap(out, dtype=DTYPE, mode="r")
        return {"file": out.name, "raw": str(path), "num_tokens": len(tokens), "num_docs": int((tokens == get_tokenizer().eos_token_id).sum()), "skipped": True}

    tok = get_tokenizer()
    eos = tok.eos_token_id
    num_tokens = num_docs = 0
    t0 = time.time()
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        for texts in read_batches(path, batch_size):
            texts = [t for t in texts if t]  # skip empty / None
            # split_special_tokens: a literal "<|endoftext|>" inside a document must not become EOS.
            ids = tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            arr = np.fromiter((i for doc in ids for i in (*doc, eos)), dtype=np.int64)
            assert arr.max(initial=0) < 2**16, "token id does not fit in uint16"
            arr.astype(DTYPE).tofile(f)  # raw values, no .npy header
            num_tokens += len(arr)
            num_docs += len(ids)
    os.replace(tmp, out)
    return {"file": out.name, "raw": str(path), "num_tokens": num_tokens, "num_docs": num_docs,
            "seconds": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[*SOURCES, "all"])
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()
    names = list(SOURCES) if args.source == "all" else [args.source]

    jobs = []
    for name in names:
        files = input_files(name)[:1]
        if not files:
            print(f"{name}: no input files in {RAW / SOURCES[name][0]}, skipping")
        (OUT / name).mkdir(parents=True, exist_ok=True)
        jobs += [(path, OUT / name, SOURCES[name][1]) for path in files]

    # Files run in parallel; results are matched back to their sources in sorted order.
    print(f"Tokenizing {len(jobs)} files with {args.workers} workers -> {OUT}/")
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(tokenize_file, jobs, chunksize=1)
    wall = time.time() - t0

    manifest = json.loads((RAW / "manifest.json").read_text())
    eos = get_tokenizer().eos_token_id
    print(f"\n{'source':12s} {'files':>5s} {'tokens':>14s} {'docs':>11s} {'GB':>6s}")
    estimates = []
    for name in names:
        files = [r for (_, out_dir, *_), r in zip(jobs, results) if out_dir.name == name]
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
        print(f"{name:12s} {len(files):5d} {meta['num_tokens']:14,} {meta['num_docs']:11,} {meta['num_tokens'] * 2 / GB:6.2f}")

        if "seconds" in files[0]:  # not measured when the output already existed
            f = files[0]
            if name == "python_edu":
                full_tokens = f["num_tokens"] / f["num_docs"] * PYTHON_EDU_DOCS
            else:  # scale by raw bytes: this whole file -> the whole dataset
                full_tokens = f["num_tokens"] / Path(f["raw"]).stat().st_size * FULL_RAW_BYTES[name]
            estimates.append((name, f["num_tokens"] / f["seconds"], full_tokens))

    print(f"\nwall time {wall:.0f}s")
    if estimates:
        print(f"\nFull-run estimate ({args.workers} workers, assuming per-file speed scales across workers):")
        print(f"{'source':12s} {'tok/s/worker':>13s} {'tokens':>10s} {'disk GB':>8s} {'time':>8s}")
        total_tokens = total_hours = 0
        for name, rate, full_tokens in estimates:
            hours = full_tokens / (rate * args.workers) / 3600
            total_tokens += full_tokens
            total_hours += hours
            print(f"{name:12s} {rate:13,.0f} {full_tokens / 1e9:9.2f}B {full_tokens * 2 / GB:8.1f} {hours:7.1f}h")
        print(f"{'total':12s} {'':>13s} {total_tokens / 1e9:9.2f}B {total_tokens * 2 / GB:8.1f} {total_hours:7.1f}h")


if __name__ == "__main__":
    main()
