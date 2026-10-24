# nano-mix

Pretraining data for a small language model, and a ~124M GPT trained on it. The first experiment
is a 5B-token mix of four sources (fineweb_edu 0.60, python_edu 0.20, finemath 0.13, gutenberg
0.07), one pass, on one H100. Raw data is kept for bigger models and new sources later.

## Layout

```
configs/mix.yaml          the training mix: sources and shares, size, seed
src/nanomix/
  paths.py                every data / output folder
  sources.py              the sources, eval-only shards, tokenizer name (one place)
  data/                   download -> tokenize -> mix, each with its verify_* check; loader.py feeds training
  model/                  config.py (ModelConfig, "40m" / "124m"), gpt.py (the model)
  train/                  cli.py (entry point), loop.py, settings.py ("mac" / "h100"), optim.py,
                          precision.py, checkpoint.py, metrics.py
  eval/                   make_eval_sets.py, bpb.py, sample.py, export_hf.py, benchmarks.py
docs/                     how and why (below)
results/                  benchmark results (in git)
data/ checkpoints/ logs/ exports/ wandb/   not in git
```

Each pipeline step reads only the previous step's output. `data/` code never imports `train/` code.

## Setup

Python is managed with uv only (see `CLAUDE.md`).

```bash
uv sync
uv run hf auth login     # read token from https://huggingface.co/settings/tokens
```

## Commands

Run every command from the repository root. Long jobs: inside `tmux`, wrapped in `caffeinate -i`
on the Mac. If Hugging Face calls hang, prefix with `HF_HUB_OFFLINE=1`.

```bash
# 1. Download raw data to data/raw/ (never modified afterwards), then check it
uv run nanomix-download --source all
uv run nanomix-verify-download [--deep]

# 2. Tokenize to data/tokenized/<source>/*.npy (flat uint16), then check
uv run nanomix-tokenize --source all
uv run nanomix-verify-tokenized

# 3. Build the mix from configs/mix.yaml, then check it (all checks must pass)
uv run nanomix-mix
uv run nanomix-verify-mix
uv run nanomix-trim-mix          # optional: a ~10 GB copy of only the tokens the mix reads

# 4. Train (presets in train/settings.py). Use tee -ia, never tee -a (CLAUDE.md).
uv run nanomix-train --preset mac --max-steps 20 --set warmup_steps=2 --tag smoke
uv run nanomix-train --model 124m --preset h100 2>&1 | tee -ia logs/train_124m.log
uv run nanomix-train --model 124m --preset h100 --resume checkpoints/<run_id>/ckpt.pt 2>&1 | tee -ia logs/train_124m.log

# 5. Evaluate
uv run nanomix-make-eval-sets    # the holdout sets are frozen: existing sets are never rebuilt
uv run nanomix-eval-bpb --checkpoint checkpoints/<run_id>/ckpt.pt
uv run nanomix-export-hf --checkpoint checkpoints/<run_id>/ckpt.pt --out exports/<run_id>
uv run nanomix-benchmarks exports/<run_id>
uv run nanomix-benchmarks --summary
```

Tests (CPU only, ~15 s): `uv run pytest`.

The run id is the mix shares, the seed and the model size, e.g. `fw60_py20_fm13_gb07_s1234_124m`.

## Docs

| doc | what |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | download, tokenize, and how OLMo-core builds the mix |
| [docs/training.md](docs/training.md) | the model, the training loop, settings, checkpoints and resume |
| [docs/eval.md](docs/eval.md) | holdout sets, BPB, export, benchmarks, results so far |
| [docs/run_on_gpu.md](docs/run_on_gpu.md) | from an empty GPU box (Vast, AWS) to a finished run |
| [docs/decisions.md](docs/decisions.md) | fixed decisions, with dates |
| [docs/roadmap.md](docs/roadmap.md) | open items |
| [docs/cleanup_and_understand.md](docs/cleanup_and_understand.md), [docs/answers.md](docs/answers.md) | the cleanup plan, and the must-know list with answers |
