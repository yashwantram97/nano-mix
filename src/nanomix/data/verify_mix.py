"""Check the mix that mix.py builds (same mix.yaml, same build code). All checks must pass.

    uv run nanomix-verify-mix [--config configs/mix.yaml]

1. Total size: instances x sequence_length >= requested_tokens, rounded up to less than one batch.
2. Ratios: each source's actual share is within 0.5 percentage points of its target.
3. No repetition: no file appears twice and no file gives more tokens than it has.
4. Content: 2 random instances per source match their source file and decode to clean text.
5. Determinism: a second build gives the same files, offsets and first 10 instances.
6. Shape: every instance has exactly sequence_length tokens, all ids < vocab size.
"""

import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from nanomix.data.mix import build_dataset, check_coverage, load_config, tokens_per_source
from nanomix.paths import MIX_CONFIG

RATIO_TOLERANCE = 0.005
SAMPLES_PER_SOURCE = 2
PREVIEW_CHARS = 300


def instance_ids(dataset, index):
    return dataset[index]["input_ids"].numpy()


def file_ranges(dataset):
    """(source, path, first instance, end instance) for every file in the mix."""
    return [(Path(p).parent.name, str(p), start, end) for p, (start, end) in zip(dataset.paths, dataset.offsets)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=MIX_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.WARNING)  # keep OLMo-core's build logs out of the report

    metas = check_coverage(cfg)
    dataset, tokenizer = build_dataset(cfg)
    seq_len = dataset.sequence_length
    dtype = np.dtype(cfg["dtype"])
    eos = tokenizer.eos_token_id
    checks = {}

    # 1. Total size. OLMo-core rounds requested_tokens up to whole global batches.
    total = len(dataset) * seq_len
    expected = math.ceil(cfg["requested_tokens"] / cfg["global_batch_size"]) * cfg["global_batch_size"]
    print(f"\n1. Total size: {len(dataset):,} instances x {seq_len} = {total:,} tokens")
    print(f"   requested {cfg['requested_tokens']:,}; rounded up to whole batches {expected:,}; "
          f"over by {total - cfg['requested_tokens']:,} ({(total / cfg['requested_tokens'] - 1) * 100:+.3f}%)")
    checks["1. total size"] = total == expected and cfg["requested_tokens"] <= total

    # 2 + 3. Ratios and no repetition, from OLMo-core's per-file instance offsets.
    used = tokens_per_source(dataset)
    ranges = file_ranges(dataset)
    paths = [p for _, p, _, _ in ranges]
    no_dup_files = len(paths) == len(set(paths))
    per_file_ok = all((end - start) * seq_len <= Path(p).stat().st_size // dtype.itemsize for _, p, start, end in ranges)
    ratio_ok = {n: abs(used[n] / total - r) <= RATIO_TOLERANCE for n, r in cfg["sources"].items()}
    reuse_ok = {n: used[n] <= metas[n]["num_tokens"] for n in cfg["sources"]}
    print(f"\n2. Ratios: see table below (tolerance +/-{RATIO_TOLERANCE * 100:.1f} percentage points)")
    print(f"3. No repetition: {len(paths)} files, each used once: {no_dup_files}; "
          f"no file gives more tokens than it has: {per_file_ok}")
    checks["2. ratios"] = all(ratio_ok.values())
    checks["3. no repetition"] = no_dup_files and per_file_ok and all(reuse_ok.values())

    # 4. Content: each sampled instance must equal the expected slice of its source file
    #    (instance k of a file = tokens [k*seq_len, (k+1)*seq_len) of that file).
    hf_tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    rng = random.Random(cfg["seed"])
    content_ok = {}
    print("\n4. Content")
    for name in cfg["sources"]:
        source_ranges = [r for r in ranges if r[0] == name and r[3] > r[2]]
        tokens_in_file = {p: np.memmap(p, dtype=dtype, mode="r") for _, p, _, _ in source_ranges}
        eos_in_used = sum(int((tokens_in_file[p][: (e - s) * seq_len] == eos).sum()) for _, p, s, e in source_ranges)
        ok = eos_in_used > 0  # documents end inside the part of the source the mix uses
        print(f"\n   {name}: {eos_in_used:,} document ends (EOS) in the {used[name]:,} tokens used")
        for i in range(SAMPLES_PER_SOURCE):
            _, path, start, end = rng.choice(source_ranges)
            index = rng.randrange(start, end)
            ids = instance_ids(dataset, index)
            local = index - start
            matches = np.array_equal(ids, tokens_in_file[path][local * seq_len : (local + 1) * seq_len])
            ok = ok and matches
            text = hf_tok.decode(ids.tolist())[:PREVIEW_CHARS]
            print(f"   --- sample {i + 1}: instance {index:,} = {Path(path).name} tokens "
                  f"{local * seq_len:,}-{(local + 1) * seq_len:,}; matches file: {matches}; EOS in instance: {int((ids == eos).sum())}")
            print("   " + text.replace("\n", "\n   "))
        content_ok[name] = ok
    checks["4. content"] = all(content_ok.values())

    # 5. Determinism: build again from the same config.
    dataset2, _ = build_dataset(cfg)
    same_layout = list(map(str, dataset2.paths)) == paths and tuple(dataset2.offsets) == tuple(dataset.offsets)
    same_first = all(np.array_equal(instance_ids(dataset, i), instance_ids(dataset2, i)) for i in range(10))
    print(f"\n5. Determinism: same files and offsets: {same_layout}; first 10 instances identical: {same_first}")
    checks["5. determinism"] = same_layout and same_first

    # 6. Shape and token ids of every instance, read through the dataset.
    bad_shape = bad_ids = 0
    for index in tqdm(range(len(dataset)), desc="6. checking every instance", unit="inst", mininterval=2):
        ids = instance_ids(dataset, index)
        bad_shape += len(ids) != seq_len
        bad_ids += int(ids.max()) >= tokenizer.vocab_size
    print(f"6. Shape: {bad_shape} instances not {seq_len} tokens long; "
          f"{bad_ids} with an id >= vocab size {tokenizer.vocab_size}")
    checks["6. shape"] = bad_shape == 0 and bad_ids == 0

    # Report.
    print(f"\n{'source':12s} {'target':>7s} {'actual':>8s} {'tokens used':>14s} {'available':>14s}  result")
    for name, ratio in cfg["sources"].items():
        ok = ratio_ok[name] and reuse_ok[name] and content_ok[name]
        print(f"{name:12s} {ratio:7.3f} {used[name] / total:8.4f} {used[name]:14,} "
              f"{metas[name]['num_tokens']:14,}  {'PASS' if ok else 'FAIL'}")
    print()
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit("\nFAILED")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
