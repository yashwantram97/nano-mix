# LLM Pretraining Data Pipeline

Building pretraining data for a ~100M-parameter model. The first experiment is a
1B-token mix. The raw data is kept for future, bigger models and new sources.

## Environment
- macOS, Python managed with **uv**. Never use pip, conda or a global Python.
- Add packages: `uv add <pkg>`. Run scripts: `uv run python <script>.py ...`
- Commit `pyproject.toml`, `uv.lock` and `.python-version`.
- No CUDA on this Mac. Data scripts must run on CPU only, with no GPU-only dependencies.
- Long jobs: run inside `tmux`, wrapped in `caffeinate -i` so the Mac doesn't sleep.
- If Hugging Face calls hang (this network's IPv6 route to huggingface.co is broken), prefix the
  command with `HF_HUB_OFFLINE=1`; the tokenizer is cached locally.

## Pipeline
1. **Download**: `download.py` saves raw data to `data/raw/<source>/`; `verify_download.py` checks it
2. **Tokenize**: `tokenize_data.py` saves token files to `data/tokenized/<source>/`; `verify_tokenized.py` checks them
3. **Mix**: `mix.py` builds the training mix from `data/tokenized/` with OLMo-core's source mixture
   API (`SourceMixtureDatasetConfig` + `NumpyFSLDatasetConfig.from_src_mix`, ai2-olmo-core 3.0.0).
   All settings are in `mix.yaml`; OLMo-core caches index files in `data/mix/`. `verify_mix.py`
   checks size, ratios (±0.5 pp), no repetition, content, determinism and instance shape.
   Builds and checks only; no training. Details in `mix.md`. `trim_mix.py` writes a copy holding
   only the tokens the mix reads (+ `manifest.json`), for boxes that download the data; `mix.py`
   uses its recorded allocation when that folder is `tokenized_dir`.
4. **Train**: `train.py --model {124m,40m}` trains `model.py`'s GPT (named configs in `config.py`
   `MODEL_CONFIGS`; 124M at 1024 context is the default, 40M at 256 was the Mac pilot) on the mix.
   `dataloader.py` builds the mix with `mix.py`'s `build_dataset`, holds out `val_instances` for
   validation, and reads each training instance at most once.
5. **Eval**: `make_eval_sets.py` builds the frozen holdout sets in `data/eval/`; `eval_bpb.py` scores
   them (called from `train.py` every `eval_every_tokens`); `export_hf.py` converts a checkpoint to
   `LlamaForCausalLM`; `run_benchmarks.py` runs lm-eval and writes `results/`. Commands in `README.md`.

```bash
uv run python mix.py          # coverage check, build, OLMo-core mix tables
uv run python verify_mix.py   # all checks must pass
```

Each step only reads the previous step's output. Never modify `data/raw/`.

## Sources
| name | repo | subset |
|---|---|---|
| fineweb_edu | HuggingFaceFW/fineweb-edu | data/CC-MAIN-2025-26 |
| finemath | HuggingFaceTB/finemath | finemath-4plus |
| python_edu | HuggingFaceTB/smollm-corpus | python-edu (code fetched from Software Heritage S3) |
| gutenberg | common-pile/project_gutenberg_filtered | all |

Planned first mix: fineweb_edu 0.60, python_edu 0.20, finemath 0.13, gutenberg 0.07.

## Fixed decisions (don't change without asking)
- Tokenizer: `HuggingFaceTB/SmolLM2-135M` (vocab 49152, fits in uint16)
- Token files: flat raw **uint16** arrays (no .npy header), one per raw input file with the
  same base name (`000_00000.parquet` -> `000_00000.npy`). Each document = tokens + EOS (id 0).
- **Holdout eval sets are frozen**: `data/eval/<name>.npy|.json` (+ the documents in
  `data/raw_eval/`), ~2M tokens each: fineweb_edu, python_edu, finemath, gutenberg, wikipedia_en.
  Never regenerate or edit them. They come from shards reserved as **eval-only**, which must never
  be downloaded or tokenized for training: fineweb_edu `000_00049.parquet`, finemath
  `train-00063-of-00064.parquet`, gutenberg `project_gutenberg-dolma-0014.json.gz`, and for python_edu
  the last 8,000 rows of `python-edu/train-00000-of-00002.parquet` (exclude those blob_ids, listed in
  `data/raw_eval/python_edu.parquet`, when fetching more python_edu). `mix.py` refuses the shards.
- **BPB** = total loss in nats / ln 2 / total UTF-8 bytes of the scored tokens, summed over the whole
  set and divided once. Bytes per token come from the tokenizer's byte-level table (special tokens
  count 0). Non-overlapping 1024-token windows (was 256 until 2026-10-16; the 40M runs' BPB used 256
  and isn't comparable), fp32 loss. 21 byte values have no SmolLM2 token and
  are dropped at tokenization, so they are not counted.
- **Benchmarks**: lm-eval **0.4.13** (pinned), 0-shot, fp32, `--device mps`, batch size 16, samples
  logged, the same settings for every model. Pilot tasks: hellaswag, piqa, arc_easy, sciq,
  lambada_openai. Answer BPB = gold-answer nats / ln 2 / UTF-8 bytes of the gold continuation,
  summed over the task, divided once. No code-execution benchmarks yet.
- Never name a script after a standard-library module (e.g. `tokenize.py`): scripts run from this
  folder, so it would be imported instead of the stdlib module and break transformers/torch imports.

## Reference links:
- https://olmo-core.readthedocs.io/en/latest/guides/data_mixing.html