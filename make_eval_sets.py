"""Frozen holdout sets for BPB evaluation: ~2M tokens per source that the training mix never reads.

Every raw file in data/raw/ is in the training mix, so the holdout comes from shards reserved as
eval-only (never download or tokenize them for training):

  fineweb_edu   last shard   data/CC-MAIN-2025-26/000_00049.parquet
  finemath      last shard   finemath-4plus/train-00063-of-00064.parquet
  gutenberg     last shard   project_gutenberg-dolma-0014.json.gz
  python_edu    the last PYTHON_EDU_ROWS rows of python-edu/train-00000-of-00002.parquet (the pilot
                uses its first 250,000), code fetched from Software Heritage S3
  wikipedia_en  wikimedia/wikipedia 20231101.en, first shard (not a training source)

Only the first documents, up to TARGET_TOKENS, are streamed; the shards are not downloaded whole.
Documents whose exact text also appears in that source's training raw files are dropped.
Tokenized exactly like tokenize_data.py: SmolLM2 tokenizer, EOS after each document, flat uint16.

Output, per set:
  data/raw_eval/<name>.parquet   the documents used (ref = row in the shard, or blob_id) and their text
  data/eval/<name>.npy           flat uint16 tokens, no header
  data/eval/<name>.json          num_tokens, num_docs, num_bytes, source files

The sets are frozen: a set whose .json exists is never rebuilt.

    uv run python make_eval_sets.py
"""

import datetime as dt
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from download import PILOT_BLOBS, SOURCES, fetch_blob, s3_client
from tokenize_data import DTYPE, TOKENIZER, get_tokenizer, read_batches

EVAL = Path("data/eval")
RAW_EVAL = Path("data/raw_eval")
TARGET_TOKENS = 2_000_000
PYTHON_EDU_ROWS = 8_000  # ~500 tokens per file, so ~4M tokens to pick 2M from
PYTHON_EDU_METADATA = Path("data/raw/python_edu/metadata/train-00000-of-00002.parquet")

# name -> (repo, revision, file). Training sources keep download.py's pinned commits.
EVAL_SHARDS = {
    "fineweb_edu": (*SOURCES["fineweb_edu"][:2], "data/CC-MAIN-2025-26/000_00049.parquet"),
    "finemath": (*SOURCES["finemath"][:2], "finemath-4plus/train-00063-of-00064.parquet"),
    "gutenberg": (*SOURCES["gutenberg"][:2], "project_gutenberg-dolma-0014.json.gz"),
    "python_edu": (*SOURCES["python_edu"][:2], "python-edu/train-00000-of-00002.parquet"),
    "wikipedia_en": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                     "20231101.en/train-00000-of-00041.parquet"),
}


def hub_docs(repo, revision, path):
    """Yield (row, text) from a dataset file on the Hub, reading only as far as the caller consumes."""
    f = HfFileSystem().open(f"datasets/{repo}@{revision}/{path}", "rb")
    if path.endswith(".parquet"):
        row = 0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield str(row), text
                row += 1
    else:  # .json.gz, one JSON document per line
        for row, line in enumerate(gzip.open(f, "rt", encoding="utf-8")):
            yield str(row), json.loads(line)["text"]


def python_edu_docs():
    """Yield (blob_id, text) for the last PYTHON_EDU_ROWS files of the local metadata shard."""
    blob_ids = pq.read_table(PYTHON_EDU_METADATA, columns=["blob_id"]).column("blob_id").to_pylist()
    start = len(blob_ids) - PYTHON_EDU_ROWS
    assert start >= PILOT_BLOBS, "eval rows overlap the pilot's first PILOT_BLOBS rows"
    blob_ids = blob_ids[start:]
    client, mode = s3_client(32, blob_ids[0])
    if client is None:
        raise SystemExit("No S3 access for python_edu (see download.py).")
    with ThreadPoolExecutor(32) as pool:
        texts = list(tqdm(pool.map(lambda b: fetch_blob(client, b), blob_ids), total=len(blob_ids), desc="  S3"))
    yield from ((b, t) for b, t in zip(blob_ids, texts) if t is not None)


def training_hashes(name):
    """sha1 of every document in this source's training raw files (from data/tokenized/<name>/meta.json)."""
    meta_path = Path("data/tokenized") / name / "meta.json"
    if not meta_path.exists():  # wikipedia_en is not a training source
        return set()
    hashes = set()
    for f in json.loads(meta_path.read_text())["files"]:
        for texts in read_batches(Path(f["raw"]), 1000):
            hashes.update(hashlib.sha1(t.encode()).digest() for t in texts if t)
    return hashes


def build(name):
    repo, revision, path = EVAL_SHARDS[name]
    print(f"\n{name}: {repo} @ {revision[:8]}  {path}")
    seen = training_hashes(name)
    docs = python_edu_docs() if name == "python_edu" else hub_docs(repo, revision, path)

    tok = get_tokenizer()
    refs, texts, tokens, num_bytes, dropped = [], [], [], 0, 0
    for ref, text in docs:
        if not text:  # skip empty / None, as tokenize_data.py does
            continue
        if hashlib.sha1(text.encode()).digest() in seen:
            dropped += 1
            continue
        # Same call as tokenize_data.py: no special tokens, a literal "<|endoftext|>" is not EOS.
        ids = tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        tokens += [*ids, tok.eos_token_id]
        refs.append(ref)
        texts.append(text)
        num_bytes += len(text.encode("utf-8"))
        if len(tokens) >= TARGET_TOKENS:
            break
    if len(tokens) < TARGET_TOKENS:
        raise SystemExit(f"{name}: only {len(tokens):,} tokens available")

    arr = np.array(tokens, dtype=np.int64)
    assert arr.max() < 2**16, "token id does not fit in uint16"
    pq.write_table(pa.table({"ref": refs, "text": texts}), RAW_EVAL / f"{name}.parquet")
    tmp = EVAL / f"{name}.npy.tmp"
    arr.astype(DTYPE).tofile(tmp)  # raw values, no .npy header
    os.replace(tmp, EVAL / f"{name}.npy")
    meta = {
        "name": name,
        "tokenizer": TOKENIZER,
        "dtype": "uint16",
        "eos_id": tok.eos_token_id,
        "num_tokens": len(arr),
        "num_docs": len(refs),
        "num_bytes": num_bytes,  # UTF-8 bytes of the document texts; EOS adds none
        "source_files": [{"repo": repo, "revision": revision, "path": path,
                          "refs": f"{refs[0]} .. {refs[-1]}", "ref_kind": "blob_id" if name == "python_edu" else "row"}],
        "dropped_in_training": dropped,
        "created": dt.date.today().isoformat(),
    }
    # Written last: the .json marks a finished, frozen set.
    (EVAL / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    RAW_EVAL.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in EVAL_SHARDS:
        if (EVAL / f"{name}.json").exists():
            print(f"{name}: frozen, already built; not rebuilding")
            meta = json.loads((EVAL / f"{name}.json").read_text())
        else:
            meta = build(name)
        rows.append(meta)
    print(f"\n{'set':13s} {'tokens':>10s} {'docs':>7s} {'bytes':>11s} {'bytes/tok':>9s} {'dropped':>7s}")
    for m in rows:
        print(f"{m['name']:13s} {m['num_tokens']:10,} {m['num_docs']:7,} {m['num_bytes']:11,} "
              f"{m['num_bytes'] / m['num_tokens']:9.2f} {m['dropped_in_training']:7,}")


if __name__ == "__main__":
    main()
