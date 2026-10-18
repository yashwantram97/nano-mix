# data-mix

Pretraining data pipeline for a ~100M-parameter model. Step 1 is `download.py`, which saves
**raw** data to `data/raw/` (no tokenization, no text processing).

`download.py` is currently a **pilot**: it downloads 1 file per source plus the first 250,000 python_edu
files from Software Heritage S3, then prints download speeds and estimates for the full run.

## Sources (pinned commits)

| source | repo @ commit | pilot file |
|---|---|---|
| fineweb_edu | `HuggingFaceFW/fineweb-edu` @ `87f09149` | first of 50 in `data/CC-MAIN-2025-26/` (44.8 GB total) |
| finemath | `HuggingFaceTB/finemath` @ `e92b25a6` | first of 64 in `finemath-4plus/` (18.4 GB total) |
| gutenberg | `common-pile/project_gutenberg_filtered` @ `3cdf6879` | first of 15 `.json.gz` (7.6 GB total) |
| python_edu | `HuggingFaceTB/smollm-corpus` @ `3ba9d605` | first metadata file in `python-edu/` + code for its first 250,000 blobs |

## Setup

Python is managed with uv (see `CLAUDE.md`). `requirements.txt` is exported from `uv.lock`.

```bash
uv sync
uv run hf auth login     # read token from https://huggingface.co/settings/tokens
```

S3 access is tried anonymously first, then with your AWS credentials (`aws configure`).
Without either, python_edu is skipped and the other sources still download.

## Pilot run

```bash
uv run python download.py --source all
```

Use `--source fineweb_edu|finemath|gutenberg|python_edu` for a single source and `--workers N`
for the number of parallel S3 fetches (default 32). Re-running skips files that are already complete.
For a long run, start it inside `tmux` wrapped in `caffeinate -i`:

```bash
tmux new -s download
caffeinate -i uv run python download.py --source all
# detach: Ctrl-b d    reattach: tmux attach -t download
```

## Output

```
data/raw/
  fineweb_edu/  finemath/  gutenberg/      first file of each, as-is
  python_edu/metadata/                     first metadata parquet, as-is
  python_edu/content/pilot-250000.parquet  blob_id, repo_name, path, score, text
  python_edu/failed_blob_ids.txt
  manifest.json                            repo, commit, files, bytes, date, status per source
```

## Evaluation

The holdout sets are already built and frozen (never rebuild them; `make_eval_sets.py` skips
existing sets). See the "Fixed decisions" in `CLAUDE.md`.

```bash
uv run python make_eval_sets.py                  # data/eval/<name>.npy|.json, ~2M tokens per set
uv run python eval_bpb.py --check-bytes          # byte table == UTF-8 bytes of the text
uv run python eval_bpb.py --random               # random init: BPB ~ uniform (sanity check)
uv run python eval_bpb.py --checkpoint model.pth # holdout BPB of a checkpoint

# Training logs holdout BPB + 3 greedy samples every eval_every_tokens (config.py) and at the end.
# 200-step sanity run on the pilot mix, evaluating every 50 steps:
HF_HUB_OFFLINE=1 uv run python -c "from config import MODEL_CONFIG, TRAIN_SETTINGS; from train import main; main(MODEL_CONFIG, TRAIN_SETTINGS.replace(max_steps=200, eval_every_tokens=409_600))"

# Export to Hugging Face (fails unless logits match the original within 1e-3)
uv run python export_hf.py --checkpoint model.pth --out exports/fw60_py20_fm13_gb07_s1234
uv run python export_hf.py --random --out exports/random_init

# lm-eval 0-shot pilot tasks -> results/<run_name>.json, results/summary.md
uv run python run_benchmarks.py exports/fw60_py20_fm13_gb07_s1234
uv run python run_benchmarks.py gpt2 HuggingFaceTB/SmolLM2-135M EleutherAI/pythia-160m
uv run python run_benchmarks.py --summary
```

Long runs: inside `tmux`, wrapped in `caffeinate -i`.
# nano-mix
