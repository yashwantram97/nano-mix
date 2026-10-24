"""Copy only the tokens the mix reads into a small folder, for GPU boxes that download the data.

The mix reads the start of every token file: a file that gives N instances gives its first
N x sequence_length tokens. This builds the mix from the full token files (configs/mix.yaml), writes
those prefixes (same file names) to --out, plus a manifest.json with the per-file allocation,
then builds the mix again from --out and checks the instances are identical. Run it where the
full token files are (the AWS box). The full files are not changed.

    uv run nanomix-trim-mix [--config configs/mix.yaml] [--out data/tokenized_trimmed]

On the GPU box, put the trimmed folder at mix.yaml's tokenized_dir (data/tokenized/): mix.py
sees manifest.json and uses the recorded allocation. A trimmed folder only fits the mix it was
cut for; after changing mix.yaml, trim again from the full files.
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nanomix.data.mix import MANIFEST, TRIM_KEYS, build_dataset, check_coverage, load_config, load_manifest
from nanomix.paths import MIX_CONFIG

SAMPLE_INSTANCES = 2000  # random instances compared, besides the first and last of every file


def write_prefix(src, dst, n_tokens, dtype):
    """Write the first n_tokens of src to dst (via a .tmp file, so a crash leaves no short file)."""
    if dst.exists() and dst.stat().st_size == n_tokens * dtype.itemsize:
        return  # done on an earlier run
    tokens = np.memmap(src, dtype=dtype, mode="r")
    if len(tokens) < n_tokens:
        raise SystemExit(f"{src} has {len(tokens):,} tokens, the mix needs {n_tokens:,}")
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        for start in range(0, n_tokens, 64 * 2**20):
            f.write(tokens[start : min(start + 64 * 2**20, n_tokens)].tobytes())
    tmp.rename(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=MIX_CONFIG)
    parser.add_argument("--out", default="data/tokenized_trimmed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    if load_manifest(cfg):
        raise SystemExit(f"{cfg['tokenized_dir']} is already trimmed; trim from the full token files")
    if out.resolve() == Path(cfg["tokenized_dir"]).resolve():
        raise SystemExit("--out must not be tokenized_dir")
    logging.basicConfig(level=logging.WARNING)  # keep OLMo-core's build logs out of the report

    print(f"Building the mix from the full token files in {cfg['tokenized_dir']}")
    metas = check_coverage(cfg)
    full, _ = build_dataset(cfg)
    seq_len, dtype = full.sequence_length, np.dtype(cfg["dtype"])

    raw_by_file = {(name, f["file"]): f["raw"] for name, meta in metas.items() for f in meta["files"]}
    files = []
    for path, (start, end) in zip(full.paths, full.offsets):
        path = Path(path)
        source = path.parent.name
        files.append({"source": source, "file": path.name, "raw": raw_by_file[(source, path.name)],
                      "tokens": (end - start) * seq_len})

    total = sum(f["tokens"] for f in files)
    print(f"\nWriting {total:,} tokens ({total * dtype.itemsize / 1e9:.1f} GB) from {len(files)} files to {out}/")
    for f in tqdm(files, desc="trimming", unit="file"):
        (out / f["source"]).mkdir(parents=True, exist_ok=True)
        write_prefix(Path(cfg["tokenized_dir"]) / f["source"] / f["file"], out / f["source"] / f["file"],
                     f["tokens"], dtype)
    for d in out.iterdir():  # leftovers from a trim for another mix would be refused by mix.py
        if d.is_dir() and d.name not in cfg["sources"]:
            shutil.rmtree(d)
    for name in cfg["sources"]:
        keep = {f["file"] for f in files if f["source"] == name}
        for p in (out / name).glob("*.npy*"):
            if p.name not in keep:
                p.unlink()
    manifest = {"mix": {k: cfg[k] for k in TRIM_KEYS}, "metas": metas, "files": files}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nBuilding the mix again from {out}/ and comparing")
    trimmed, _ = build_dataset({**cfg, "tokenized_dir": str(out)})
    same_layout = (len(trimmed) == len(full) and tuple(trimmed.offsets) == tuple(full.offsets)
                   and [Path(p).relative_to(out.resolve()) for p in trimmed.paths]
                   == [Path(f["source"]) / f["file"] for f in files])
    print(f"same instances per file, in the same order: {same_layout}")
    if not same_layout:
        raise SystemExit("FAILED: the trimmed mix has a different layout; don't use it")

    indices = {i for start, end in full.offsets if end > start for i in (start, end - 1)}
    indices |= set(random.Random(cfg["seed"]).sample(range(len(full)), min(SAMPLE_INSTANCES, len(full))))
    bad = [i for i in tqdm(sorted(indices), desc="comparing instances", unit="inst", mininterval=2)
           if not np.array_equal(full[i]["input_ids"].numpy(), trimmed[i]["input_ids"].numpy())]
    print(f"{len(indices):,} instances compared (first and last of every file + {SAMPLE_INSTANCES:,} random): "
          f"{len(bad)} differ")
    if bad:
        raise SystemExit(f"FAILED: instances {bad[:10]} differ; don't use the trimmed folder")

    print(f"\nPASS. {out}/ gives the same {len(full):,} instances as {cfg['tokenized_dir']}/.")
    print(f"Next: uv run nanomix-verify-mix on the box that uses it, with {out}/ at {cfg['tokenized_dir']}/.")


if __name__ == "__main__":
    main()
