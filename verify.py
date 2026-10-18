"""Check the token files in data/tokenized/<source>/ against their meta.json.

    uv run python verify.py
"""

import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

SLICE = 300
GB = 1e9


def verify(source_dir, tok):
    meta = json.loads((source_dir / "meta.json").read_text())
    files = [np.memmap(source_dir / f["file"], dtype=meta["dtype"], mode="r") for f in meta["files"]]
    nbytes = sum((source_dir / f["file"]).stat().st_size for f in meta["files"])

    checks = {
        "bytes/2 == num_tokens": nbytes // 2 == meta["num_tokens"] and nbytes % 2 == 0,
        "max id < vocab": max(int(t.max()) for t in files) < len(tok),
        "EOS count == num_docs": sum(int((t == meta["eos_id"]).sum()) for t in files) == meta["num_docs"],
    }
    print(f"\n=== {source_dir.name}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # Show 3 random 300-token slices; they should read as clean text.
    rng = random.Random(0)
    for i in range(3):
        tokens = rng.choice(files)
        start = rng.randrange(max(1, len(tokens) - SLICE))
        text = tok.decode(tokens[start : start + SLICE].tolist())
        print(f"\n  --- sample {i + 1} ({source_dir.name}, tokens {start:,}-{start + SLICE:,}) ---")
        print("  " + text.replace("\n", "\n  "))

    return [source_dir.name, len(files), meta["num_tokens"], meta["num_docs"], nbytes / GB, sum(checks.values()), len(checks)]


def main():
    root = Path("data/tokenized")

    sources = sorted(d for d in root.iterdir() if (d / "meta.json").exists())
    tok = AutoTokenizer.from_pretrained(json.loads((sources[0] / "meta.json").read_text())["tokenizer"])
    rows = [verify(d, tok) for d in sources]

    print(f"\n{'source':12s} {'files':>5s} {'tokens':>14s} {'docs':>11s} {'GB':>6s}  checks")
    for name, n, tokens, docs, gb, passed, total in rows:
        print(f"{name:12s} {n:5d} {tokens:14,} {docs:11,} {gb:6.2f}  {passed}/{total}")
    if any(passed < total for *_, passed, total in rows):
        raise SystemExit("\nFAILED")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
