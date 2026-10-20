"""Build the training mix from data/tokenized/ with OLMo-core's source_mixture API. No training.

Settings are in mix.yaml.

    uv run python mix.py [--config mix.yaml]

tokenized_dir can also be a trimmed folder made by trim_mix.py (it has a manifest.json): only the
tokens this mix reads, with the per-file allocation taken from the manifest, so the instances and
their order are the same as from the full folder.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from olmo_core.data import NumpyDatasetDType, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDataset,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
    SourceMixtureOutcome,
    SourcePathTokens,
)

from checkpointing import MIX_KEYS
from download import EVAL_ONLY

MANIFEST = "manifest.json"
# mix.yaml keys a trimmed folder was cut for: the ones that decide the instances, plus the token dtype.
TRIM_KEYS = MIX_KEYS + ("dtype",)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    ratios = sum(cfg["sources"].values())
    if abs(ratios - 1.0) > 1e-9:
        raise SystemExit(f"Source ratios in {path} sum to {ratios}, not 1.0")
    if cfg["global_batch_size"] % cfg["sequence_length"]:
        raise SystemExit("global_batch_size must be a multiple of sequence_length")
    return cfg


def source_dir(cfg, name):
    return Path(cfg["tokenized_dir"]).resolve() / name


def load_manifest(cfg):
    """tokenized_dir's manifest.json if it is a trimmed folder (trim_mix.py), else None."""
    path = Path(cfg["tokenized_dir"]) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def check_not_eval(name, raw_files):
    """Stops if any raw file is a shard reserved for the holdout set.

    Shards reserved for the holdout set must never be trained on: the fixed list in download.py,
    plus data/eval/<name>.json (make_eval_sets.py) where it exists. (python_edu's holdout is a set
    of blob_ids, not a whole shard; download.py never fetches them. See CLAUDE.md.)
    """
    eval_shards = {EVAL_ONLY[name]} if name in EVAL_ONLY else set()
    eval_meta = Path("data/eval") / f"{name}.json"
    if eval_meta.exists():
        eval_shards |= {Path(s["path"]).name for s in json.loads(eval_meta.read_text())["source_files"]
                        if s["ref_kind"] == "row"}
    if eval_shards & {Path(f).name for f in raw_files}:
        raise SystemExit(f"{name}: {eval_shards} is reserved for the holdout set; remove it from the mix")


def check_trimmed(cfg, manifest):
    """Step 1 for a trimmed folder: it was cut for this mix and every file is there, whole."""
    changed = [k for k in TRIM_KEYS if manifest["mix"].get(k) != cfg.get(k)]
    if changed:
        raise SystemExit(f"{cfg['tokenized_dir']} was trimmed for a different mix ({', '.join(changed)} "
                         f"differ from mix.yaml). Trim again from the full token files.")
    itemsize = np.dtype(cfg["dtype"]).itemsize
    print(f"{cfg['tokenized_dir']} is trimmed to this mix ({MANIFEST}); allocation taken from it")
    print(f"{'source':12s} {'files':>6s} {'tokens':>14s}")
    for name in cfg["sources"]:
        files = [f for f in manifest["files"] if f["source"] == name]
        on_disk = sorted(p.name for p in source_dir(cfg, name).glob("*.npy"))
        if on_disk != sorted(f["file"] for f in files):
            raise SystemExit(f"{name}: .npy files on disk {on_disk} don't match {MANIFEST}")
        for f in files:
            size = (source_dir(cfg, name) / f["file"]).stat().st_size
            if size != f["tokens"] * itemsize:
                raise SystemExit(f"{name}/{f['file']}: {size:,} bytes, expected {f['tokens'] * itemsize:,} "
                                 f"(incomplete download?)")
        check_not_eval(name, [f["raw"] for f in files])
        print(f"{name:12s} {len(files):6d} {sum(f['tokens'] for f in files):14,}")
    return manifest["metas"]


def check_coverage(cfg):
    """Step 1: tokens available (from meta.json) vs tokens needed. Stops if any source is too small."""
    manifest = load_manifest(cfg)
    if manifest:
        return check_trimmed(cfg, manifest)
    metas, short = {}, []
    print(f"{'source':12s} {'available':>14s} {'needed':>14s} {'coverage':>9s}")
    for name, ratio in cfg["sources"].items():
        meta = json.loads((source_dir(cfg, name) / "meta.json").read_text())
        # Every .npy in the folder is globbed into the mix, so it must be exactly the files meta.json
        # describes (e.g. a leftover older pilot file would duplicate documents).
        on_disk = sorted(p.name for p in source_dir(cfg, name).glob("*.npy"))
        if on_disk != sorted(f["file"] for f in meta["files"]):
            raise SystemExit(f"{name}: .npy files on disk {on_disk} don't match meta.json")
        check_not_eval(name, [f["raw"] for f in meta["files"]])

        needed = int(cfg["requested_tokens"] * ratio)
        coverage = meta["num_tokens"] / needed
        print(f"{name:12s} {meta['num_tokens']:14,} {needed:14,} {coverage:9.2f}")
        if coverage < 1.0:
            short.append(name)
        metas[name] = meta
    if short:
        raise SystemExit(f"\nNot enough tokens for: {', '.join(short)}. Stopping (no repetition allowed).")
    return metas


@dataclass
class FixedSourceMixture(SourceMixtureDatasetConfig):
    """A source mixture whose per-file token counts are given, not computed from file sizes.

    For a trimmed folder: OLMo-core's rounding, run again on the trimmed file sizes, could give a
    different allocation, so the one recorded from the full folder is used as is.
    """

    allocation: list = field(default_factory=list)  # [(source, path, tokens)] in the original order

    def build(self, *, npdtype, sequence_length):
        sources = {}
        for source, path, tokens in self.allocation:
            sources.setdefault(source, []).append(SourcePathTokens(path=path, tokens=tokens, max_tokens=tokens))
        return SourceMixtureDataset(sources=[SourceMixtureOutcome(name=n, path_tokens=p) for n, p in sources.items()])


def build_dataset(cfg):
    """Build the mixed OLMo-core dataset described by cfg. Returns (dataset, tokenizer config)."""
    tokenizer = TokenizerConfig.from_hf(cfg["tokenizer"])
    manifest = load_manifest(cfg)
    mixture_cls = FixedSourceMixture if manifest else SourceMixtureDatasetConfig
    extra = {"allocation": [(f["source"], str(source_dir(cfg, f["source"]) / f["file"]), f["tokens"])
                            for f in manifest["files"]]} if manifest else {}
    src_mix = mixture_cls(
        **extra,
        source_list=SourceMixtureList(
            sources=[
                SourceMixtureConfig(
                    source_name=name,
                    target_ratio=ratio,
                    paths=[str(source_dir(cfg, name) / "*.npy")],
                )
                for name, ratio in cfg["sources"].items()
            ]
        ),
        requested_tokens=cfg["requested_tokens"],
        global_batch_size=cfg["global_batch_size"],
        seed=cfg["seed"],
    )
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix,
        tokenizer=tokenizer,
        sequence_length=cfg["sequence_length"],
        dtype=NumpyDatasetDType(cfg["dtype"]),
        work_dir=cfg["work_dir"],
    )
    dataset = dataset_config.build()  # picks how many tokens to take from each file
    dataset.prepare()  # writes OLMo-core's index files into work_dir
    return dataset, tokenizer


def tokens_per_source(dataset):
    """Tokens the built dataset uses from each source, from OLMo-core's per-file instance offsets."""
    used = {}
    for path, (start, end) in zip(dataset.paths, dataset.offsets):
        source = Path(path).parent.name
        used[source] = used.get(source, 0) + (end - start) * dataset.sequence_length
    return used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mix.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("Step 1: coverage")
    metas = check_coverage(cfg)

    print("\nStep 2: build")
    tokenizer = TokenizerConfig.from_hf(cfg["tokenizer"])
    print(f"tokenizer {cfg['tokenizer']}: vocab_size={tokenizer.vocab_size}, eos_token_id={tokenizer.eos_token_id}")
    for name, meta in metas.items():
        assert meta["eos_id"] == tokenizer.eos_token_id, f"{name}: meta.json eos_id {meta['eos_id']} != {tokenizer.eos_token_id}"

    logging.basicConfig(level=logging.INFO, format="%(message)s")  # OLMo-core logs its mix tables
    dataset, _ = build_dataset(cfg)

    used = tokens_per_source(dataset)
    total = len(dataset) * dataset.sequence_length
    print(f"\nInstances: {len(dataset):,} x {dataset.sequence_length} = {total:,} tokens (requested {cfg['requested_tokens']:,})")
    print(f"{'source':12s} {'target':>7s} {'actual':>7s} {'tokens used':>14s} {'available':>14s}")
    for name, ratio in cfg["sources"].items():
        print(f"{name:12s} {ratio:7.3f} {used[name] / total:7.3f} {used[name]:14,} {metas[name]['num_tokens']:14,}")


if __name__ == "__main__":
    main()
