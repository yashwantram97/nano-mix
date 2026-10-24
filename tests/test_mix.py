"""A full and a trimmed build of the same mix give the same instances (trim_mix.py, mix.py)."""

import json
import sys

import numpy as np
import pytest
import yaml

from nanomix.data import trim_mix
from nanomix.data.mix import build_dataset, load_config
from nanomix.sources import TOKENIZER

SEQ_LEN = 64


def write_source(root, name, file_tokens, rng):
    """Token files + meta.json as tokenize_data.py writes them: flat uint16, EOS (0) after each doc."""
    folder = root / name
    folder.mkdir(parents=True)
    files = []
    for i, n in enumerate(file_tokens):
        tokens = rng.integers(1, 49152, n, dtype=np.uint16)
        tokens[rng.choice(n, n // 200, replace=False)] = 0  # document ends
        tokens.tofile(folder / f"{i:03d}.npy")
        files.append({"file": f"{i:03d}.npy", "raw": f"data/raw/{name}/{i:03d}.parquet",
                      "num_tokens": n, "num_docs": int((tokens == 0).sum())})
    meta = {"tokenizer": TOKENIZER, "dtype": "uint16", "eos_id": 0,
            "num_tokens": sum(file_tokens), "num_docs": sum(f["num_docs"] for f in files), "files": files}
    (folder / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def mix_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mix.py looks for data/eval/ relative to the working directory
    rng = np.random.default_rng(0)
    write_source(tmp_path / "tokenized", "fineweb_edu", [9_000, 7_000, 11_000], rng)
    write_source(tmp_path / "tokenized", "python_edu", [6_000, 8_000], rng)
    cfg = {"requested_tokens": 16_384, "sequence_length": SEQ_LEN, "global_batch_size": 4 * SEQ_LEN,
           "seed": 1234, "tokenizer": TOKENIZER, "dtype": "uint16",
           "tokenized_dir": str(tmp_path / "tokenized"), "work_dir": str(tmp_path / "mix"),
           "sources": {"fineweb_edu": 0.6, "python_edu": 0.4}}
    path = tmp_path / "mix.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_trimmed_mix_gives_the_same_instances(tmp_path, mix_config, monkeypatch):
    out = tmp_path / "trimmed"
    monkeypatch.setattr(sys, "argv", ["trim_mix", "--config", str(mix_config), "--out", str(out)])
    trim_mix.main()  # also compares the two builds itself

    cfg = load_config(mix_config)
    full, _ = build_dataset(cfg)
    trimmed, _ = build_dataset({**cfg, "tokenized_dir": str(out), "work_dir": str(tmp_path / "mix_trimmed")})
    assert len(full) == len(trimmed) > 0
    for i in range(len(full)):
        assert np.array_equal(full[i]["input_ids"].numpy(), trimmed[i]["input_ids"].numpy()), i
    # The copy holds only what the mix reads.
    trimmed_tokens = sum(p.stat().st_size for p in out.rglob("*.npy")) // 2
    assert trimmed_tokens == len(full) * SEQ_LEN


def test_mix_refuses_another_tokenizer(tmp_path, mix_config):
    cfg = yaml.safe_load(mix_config.read_text())
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({**cfg, "tokenizer": "gpt2"}))
    with pytest.raises(SystemExit, match="tokenizer"):
        load_config(other)
