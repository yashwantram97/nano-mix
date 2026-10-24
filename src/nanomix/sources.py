"""The data sources, the eval-only shards and the tokenizer: one place for these facts.

download.py, tokenize_data.py, make_eval_sets.py, mix.py and verify_download.py all import them
from here. configs/mix.yaml repeats the tokenizer name (it is part of the recorded mix), and
mix.py checks it equals TOKENIZER.
"""

from dataclasses import dataclass

import numpy as np

from nanomix.paths import RAW_EVAL

# HuggingFaceTB/SmolLM2-135M: byte-level BPE, vocab 49,152, EOS id 0, no BOS (CLAUDE.md).
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
DTYPE = np.uint16  # 49,152 ids fit in 16 bits


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str  # pinned commit, so the download is reproducible
    pattern: str  # files of the subset in the repo
    raw_glob: str  # input files under data/raw/ for tokenize_data.py
    batch_docs: int = 1000  # documents tokenized per batch (Gutenberg books are huge)
    abbrev: str = ""  # short name in run ids (fw60_py20_...)


SOURCES = {
    "fineweb_edu": Source("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                          "data/CC-MAIN-2025-26/*.parquet", "fineweb_edu/**/*.parquet", abbrev="fw"),
    "finemath": Source("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594",
                       "finemath-4plus/*.parquet", "finemath/**/*.parquet", abbrev="fm"),
    "gutenberg": Source("common-pile/project_gutenberg_filtered", "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7",
                        "*.json.gz", "gutenberg/**/*.json.gz", batch_docs=8, abbrev="gb"),
    # Metadata from HF; the code itself is fetched from Software Heritage S3 (download.py).
    "python_edu": Source("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f",
                         "python-edu/*.parquet", "python_edu/content/*.parquet", abbrev="py"),
}

# Shards reserved for the frozen holdout sets (CLAUDE.md). Never downloaded or tokenized for
# training; mix.py refuses them.
EVAL_ONLY = {
    "fineweb_edu": "000_00049.parquet",
    "finemath": "train-00063-of-00064.parquet",
    "gutenberg": "project_gutenberg-dolma-0014.json.gz",
}

# python_edu's holdout is the last 8,000 rows of this metadata file; the blob_ids it used are in
# data/raw_eval/python_edu.parquet. Those rows are never fetched, and their blob_ids are skipped anywhere.
PYTHON_EDU_EVAL_FILE, PYTHON_EDU_EVAL_ROWS = "train-00000-of-00002.parquet", 8_000
PYTHON_EDU_EVAL_BLOBS = RAW_EVAL / "python_edu.parquet"
PILOT_CONTENT = "pilot-250000.parquet"  # the pilot's rows 0-250k of PYTHON_EDU_EVAL_FILE, reused as that chunk
PILOT_ROWS = 250_000
PILOT_DUPLICATE = "pilot-5000.parquet"  # an earlier pilot: the first 5,000 rows of PILOT_CONTENT again

# Holdout sets: name -> (repo, revision, file). Training sources keep their pinned commits;
# wikipedia_en is not a training source.
EVAL_SHARDS = {
    **{name: (SOURCES[name].repo, SOURCES[name].revision, path) for name, path in {
        "fineweb_edu": "data/CC-MAIN-2025-26/" + EVAL_ONLY["fineweb_edu"],
        "finemath": "finemath-4plus/" + EVAL_ONLY["finemath"],
        "gutenberg": EVAL_ONLY["gutenberg"],
        "python_edu": "python-edu/" + PYTHON_EDU_EVAL_FILE,
    }.items()},
    "wikipedia_en": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                     "20231101.en/train-00000-of-00041.parquet"),
}
