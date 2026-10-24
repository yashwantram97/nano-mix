# nano-mix: LLM pretraining data pipeline

Pretraining data for a small model and a ~124M GPT trained on it (first run: a 5B-token mix, one
H100). The raw data is kept for future, bigger models and new sources. Layout and commands:
`README.md`. The how and why: `docs/` (pipeline, training, eval, run_on_gpu, decisions, roadmap).

## Environment
- macOS, Python managed with **uv**. Never use pip, conda or a global Python.
- Add packages: `uv add <pkg>`. Run: `uv run nanomix-<step> ...` or `uv run python -m nanomix.<module>`,
  from the repository root (paths are relative to it, see `src/nanomix/paths.py`).
- Commit `pyproject.toml`, `uv.lock` and `.python-version`.
- Tests: `uv run pytest` (`tests/`, CPU only, small). Run them after every change to `src/`.
- No CUDA on this Mac. Data code must run on CPU only, with no GPU-only dependencies.
- Long jobs: run inside `tmux`, wrapped in `caffeinate -i` so the Mac doesn't sleep.
- Logging with a pipe: use `tee -ia`, never `tee -a`. Ctrl-C reaches every process in the pipeline,
  so a plain `tee` exits at once; the trainer's next print then hits a broken pipe (BrokenPipeError)
  and it dies before the checkpoint save, and the log just stops with no traceback. (Lost the
  smoke test's final save, 2026-10-21.) `-i` only covers Ctrl-C: a killed tmux session (SIGHUP) or a
  container stop (SIGTERM) still kills `tee`.
- If Hugging Face calls hang (this network's IPv6 route to huggingface.co is broken), prefix the
  command with `HF_HUB_OFFLINE=1`; the tokenizer is cached locally.

## Code rules
- Each pipeline step reads only the previous step's output: download → tokenize → mix → train → eval.
- `nanomix.data` never imports `nanomix.train`.
- One place for each fact: sources, eval-only shards and the tokenizer in `sources.py`; folders in
  `paths.py`; the mix in `configs/mix.yaml`. Import them; don't repeat them.
- Never name a module after a standard-library module (e.g. `tokenize.py`).
- Checkpoints hold only tensors and plain values. Don't rename `TrainSettings` / `ModelConfig`
  fields or `MIX_KEYS`: a resume compares them with the checkpoint's.

## Fixed decisions (don't change without asking; reasons and dates in `docs/decisions.md`)
- Tokenizer `HuggingFaceTB/SmolLM2-135M` (vocab 49152, uint16). Token files: flat raw uint16, no
  header, one per raw file with the same base name; each document = tokens + EOS (id 0).
- Never modify `data/raw/`.
- **Holdout eval sets are frozen**: `data/eval/` and `data/raw_eval/`. Never regenerate or edit them.
- **Eval-only shards** (`sources.py` `EVAL_ONLY`, and the last 8,000 rows of python_edu's
  `train-00000-of-00002.parquet`) are never downloaded or tokenized for training.
- BPB: total nats / ln 2 / total UTF-8 bytes, summed over the set and divided once; non-overlapping
  1024-token windows; fp32.
- Benchmarks: lm-eval 0.4.13, 0-shot, fp32, `--device mps`, batch 16, the same for every model.
- Mix: fineweb_edu 0.60, python_edu 0.20, finemath 0.13, gutenberg 0.07; no repetition.

## Reference links
- https://olmo-core.readthedocs.io/en/latest/guides/data_mixing.html
