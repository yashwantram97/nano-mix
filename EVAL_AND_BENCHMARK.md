# Evaluation and benchmarks

How the pretraining experiments are evaluated: frozen holdout sets scored in bits per byte (BPB)
during training, export to Hugging Face, and 0-shot lm-eval benchmarks. This is the **pilot**
version: pilot task set only, and training metrics go to the terminal (no TensorBoard yet).
The fixed decisions are also listed in `CLAUDE.md`.

## Contents

1. [Files](#files)
2. [Holdout eval sets](#1-holdout-eval-sets-make_eval_setspy)
3. [BPB evaluation](#2-bpb-evaluation-eval_bpbpy)
4. [Training integration and logging](#3-training-integration-and-logging-trainpy-configpy)
5. [Export to Hugging Face](#4-export-to-hugging-face-export_hfpy)
6. [Benchmarks](#5-benchmarks-run_benchmarkspy)
7. [Sanity checks and results](#6-sanity-checks-and-results)
8. [Commands](#commands)
9. [Not done yet](#not-done-yet)

## Files

| file | status | what it does |
|---|---|---|
| `make_eval_sets.py` | new | builds the frozen holdout sets in `data/eval/` |
| `eval_bpb.py` | new | `evaluate_bpb()`, byte-length table, byte check, random-init sanity check |
| `export_hf.py` | new | GPTModel checkpoint -> `LlamaForCausalLM`, with a logit check |
| `run_benchmarks.py` | new | lm-eval 0-shot pilot tasks, answer BPB, `results/summary.md` |
| `train.py` | edited | holdout BPB + samples every `eval_every_tokens`, extra terminal metrics, run name/config |
| `config.py` | edited | new setting `eval_every_tokens = 25_000_000` |
| `mix.py` | edited | `check_coverage` refuses eval-only shards |
| `pyproject.toml`, `uv.lock` | edited | `lm-eval[hf]==0.4.13` (pulls in `accelerate`; torch/transformers unchanged) |
| `requirements.txt` | re-exported | `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt` (141 packages) |
| `CLAUDE.md`, `README.md` | edited | eval step and fixed decisions; eval commands |

Outputs:

```
data/eval/<name>.npy, <name>.json          frozen token streams + metadata (read-only)
data/raw_eval/<name>.parquet               the documents behind each set (read-only)
exports/<run_name>/                        LlamaForCausalLM + SmolLM2 tokenizer
results/<run_name>.json                    benchmark metrics per model
results/samples/<run_name>/<task>.jsonl    lm-eval logged samples
results/summary.md                         one column per model
logs/                                      output of the sanity runs below
```

## 1. Holdout eval sets (`make_eval_sets.py`)

### Where the data comes from

The pilot has exactly one raw file per source, and `mix.py` globs every token file into the mix,
so **every raw file was already in the training mix** (`python_edu/content/pilot-5000.parquet` is
not tokenized, but all 5,000 of its blobs are also in `pilot-250000.parquet`). The mix only reads a
prefix of each file, but the unused tail is the same raw file, and the line moves whenever
`requested_tokens` changes. So the holdout comes from **new shards reserved as eval-only**:

| set | repo @ commit | eval-only data |
|---|---|---|
| fineweb_edu | `HuggingFaceFW/fineweb-edu` @ `87f09149` | last shard `data/CC-MAIN-2025-26/000_00049.parquet` |
| finemath | `HuggingFaceTB/finemath` @ `e92b25a6` | last shard `finemath-4plus/train-00063-of-00064.parquet` |
| gutenberg | `common-pile/project_gutenberg_filtered` @ `3cdf6879` | last shard `project_gutenberg-dolma-0014.json.gz` |
| python_edu | `HuggingFaceTB/smollm-corpus` @ `3ba9d605` | last 8,000 rows of `python-edu/train-00000-of-00002.parquet` (the pilot uses the first 250,000), code from Software Heritage S3 |
| wikipedia_en | `wikimedia/wikipedia` @ `b04c8d1c` | `20231101.en/train-00000-of-00041.parquet` (not a training source) |

- Only the first documents, up to 2M tokens, are streamed with `HfFileSystem`; the shards are not
  downloaded whole.
- Any document whose exact text (sha1) appears in that source's training raw files is dropped.
  None were found.
- **These shards must never be downloaded or tokenized for training.** `mix.py` refuses a mix that
  contains one. For python_edu the holdout is a set of blob_ids (listed in
  `data/raw_eval/python_edu.parquet`), not a whole shard, so a future full python_edu download must
  exclude those blob_ids itself.

### Tokenization and format

Exactly as `tokenize_data.py`: SmolLM2 tokenizer, `add_special_tokens=False`,
`split_special_tokens=True`, EOS (id 0) after each document, flat uint16 with no `.npy` header.
The `.json` holds `num_tokens`, `num_docs`, `num_bytes` (UTF-8 bytes of the texts), the source
file with the range of rows/blob_ids used, and the creation date.

| set | tokens | docs | bytes | bytes/token |
|---|---|---|---|---|
| fineweb_edu | 2,001,952 | 1,886 | 9,816,803 | 4.90 |
| finemath | 2,036,328 | 1,263 | 6,340,592 | 3.11 |
| gutenberg | 2,073,354 | 15 | 8,083,239 | 3.90 |
| python_edu | 2,002,120 | 3,893 | 6,341,373 | 3.17 |
| wikipedia_en | 2,007,692 | 303 | 8,611,044 | 4.29 |

### Frozen

A set whose `.json` exists is never rebuilt (the `.json` is written last, so it marks a finished
set). The files in `data/eval/` and `data/raw_eval/` are read-only (`chmod a-w`). Never regenerate
or edit them: every BPB number is only comparable on the same sets.

## 2. BPB evaluation (`eval_bpb.py`)

`evaluate_bpb(model, eval_sets, batch_size) -> {name: bpb}`

- **Windows**: each stream is cut into non-overlapping `BPB_WINDOW` (1024) windows, fixed in
  `eval_bpb.py` rather than taken from the model, and models with a shorter context are refused.
  It was 256 until 2026-10-16, so the 40M runs' BPB isn't comparable. Window *i* reads
  tokens `[i·L, (i+1)·L)` and is scored on `[i·L+1, (i+1)·L+1)`, so every token after the first is
  scored exactly once.
- **Loss**: `torch.no_grad()`, eval mode (the previous mode is restored), padded vocab rows
  dropped, cross-entropy summed in nats, in **fp32** (no autocast) so BPB is comparable across runs.
- **BPB** = total nats / ln 2 / total UTF-8 bytes of the scored tokens, summed over the whole set
  and divided once (no per-window averaging).
- **Bytes per token** come from the tokenizer's byte-level alphabet (`bytes_to_unicode`): each
  token's byte length is its number of alphabet characters, so a token holding part of a
  multi-byte UTF-8 character counts only its own bytes. EOS and the other 16 special tokens
  (ids 0–16) count 0.

### Byte check (`--check-bytes`)

For the first 100 documents of each set (gutenberg has only 15), and for each whole set, the summed
byte table must equal `len(text.encode("utf-8"))`. **Finding:** the SmolLM2 vocab has no token for
21 byte values (`0x04 0x06 0x13 0x14 0x16 0x1d 0xc0 0xc1 0xf1 0xf2 0xf5`–`0xff`). The tokenizer
silently drops them, in the training data too, so they are never predicted and not counted. Only
finemath contains any: 9 of its 6,340,592 bytes. The check accounts for these bytes explicitly and
prints them:

```
  bytes with no token in the vocab: 0x04 0x06 0x13 0x14 0x16 0x1d 0xc0 0xc1 0xf1 0xf2 0xf5 ... 0xff
  PASS  finemath      100 docs, 0 mismatched; whole set 6,340,583 bytes = 6,340,592 in .json - 9 unencodable
  PASS  fineweb_edu   100 docs, 0 mismatched; whole set 9,816,803 bytes = 9,816,803 in .json - 0 unencodable
  PASS  gutenberg     15 docs, 0 mismatched; whole set 8,083,239 bytes = 8,083,239 in .json - 0 unencodable
  PASS  python_edu    100 docs, 0 mismatched; whole set 6,341,373 bytes = 6,341,373 in .json - 0 unencodable
  PASS  wikipedia_en  100 docs, 0 mismatched; whole set 8,611,044 bytes = 8,611,044 in .json - 0 unencodable
```

### Cost

One full eval scores about 10M tokens (5 sets × 2M) and takes about 5–6 minutes on this Mac (MPS,
fp32, batch 16). At the default interval of 25M training tokens that is roughly 13% overhead on a
full pass over the 500M mix (about 20 evals). bf16 would be about 2× faster but less exact.

## 3. Training integration and logging (`train.py`, `config.py`)

Your training code is extended, not rewritten. TensorBoard is not set up; everything goes to the
terminal.

- **Holdout eval**: every `eval_every_tokens` (default 25M, in `config.py`) and at the end of the
  run, `evaluate_bpb` on all sets in `data/eval/`, printed as one line:
  `[eval 1,632,000 tokens] bpb/finemath 2.7455  bpb/fineweb_edu 1.9655 ...`
- **Samples**: at each holdout eval, greedy 100-token generations from 3 fixed prompts (prose, code,
  math), through your `generate_text_simple`, which masks the padded vocab with `mask_padded_logits`.
- **Step line** (every `log_freq` steps): mean train loss, lr, mean grad norm *before* clipping
  (from `clip_grad_norm_`; measured even when clipping is off), clip fraction, tokens/sec, s/step,
  MFU (tok/s × `training_flops_per_token` ÷ the GPU's peak, looked up in `train.py` `PEAK_TFLOPS`:
  989 TFLOP/s for the SXM H100 in p5 instances, 756 for PCIe) and peak CUDA memory. Losses and grad
  norms stay on the GPU until this line, so it is the only host sync. Throughput covers training
  only: time spent on val loss, sampling and holdout evals is excluded.
- **Val line** (every `eval_freq` steps): train loss (mean since the last val line, no extra forward
  passes), val loss on `eval_iter` fixed val batches loaded once at the start, and param norm.
- **Metrics file**: every train, val and BPB record is appended as one JSON line to
  `logs/<run_name>.jsonl` (`metrics_dir`), after a `run` record with the full config.
- **Run name and config**: printed once at the start, e.g. `Run: fw60_py20_fm13_gb07_s1234_124m`
  (mix ratios in percent, the mix seed, the model size), followed by the mix ratios, seed, requested
  tokens, model size and the full model and training settings as JSON.
- **Presets**: `train.py --preset mac` (quick test, the `TrainSettings` defaults) or `--preset h100`
  (`H100_SETTINGS`, the 124M / 5B run). `resolve_device()` was pulled out of `main()` so
  `eval_bpb.py` can reuse it.

## 4. Export to Hugging Face (`export_hf.py`)

Converts a GPTModel state dict to `LlamaForCausalLM` and saves it with the SmolLM2 tokenizer.

| GPTModel | Llama |
|---|---|
| `token_emb` / `out_head` (tied) | `embed_tokens` / `lm_head`, `tie_word_embeddings` from `config.py` |
| `norm1`, `norm2`, `final_norm` (`nn.RMSNorm`, eps 1e-5) | `input_layernorm`, `post_attention_layernorm`, `norm` |
| `W_query`, `W_key`, `W_value`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `w_gate`, `w_up`, `w_down` (SwiGLU) | `gate_proj`, `up_proj`, `down_proj` |
| RoPE, rotate-half, base 10000 | `rope_parameters = {"rope_type": "default", "rope_theta": 10000}` |

- **One difference from "no biases"**: `model.py`'s attention output projection is an
  `nn.Linear` with its default bias. Llama's `attention_bias` turns on all four q/k/v/o biases
  together, so the export sets q/k/v biases to zero and keeps the trained `o_proj` bias. The forward
  pass is identical.
- `max_position_embeddings` = the training context (256 for 40M, 1024 for 124M), so lm-eval
  truncates longer contexts from the left.
- **Verification**: the saved files are reloaded and run on random tokens (2 × `context_length`) next to the
  original model, both fp32 on CPU. The script fails unless max |logit difference| < 1e-3. On
  `model.pth` the difference is exactly 0 (both run the same fp32 kernels). To confirm the check
  can fail: zeroing one `o_proj` bias gives 0.035, and a wrong RoPE theta gives 2.18.

## 5. Benchmarks (`run_benchmarks.py`)

- **lm-eval 0.4.13**, pinned as `lm-eval[hf]==0.4.13` (the `hf` extra adds `accelerate`, which its
  HF backend needs).
- **Pilot tasks**: hellaswag, piqa, arc_easy, sciq, lambada_openai.
- **The same settings for every model**: 0-shot, fp32, device `mps`, batch size 16, samples logged,
  lm-eval's default seeds. Run through `lm_eval.simple_evaluate`.
- Works for exported checkpoints (a directory; the run name is the directory name) and HF models by
  name (the run name replaces `/` with `__`).
- **Answer BPB** from the logged samples: the gold answer's log-likelihood in nats / ln 2 / UTF-8
  bytes of the gold continuation (including its leading space), summed over the task and divided
  once. For multiple-choice tasks the gold is `arguments[target]`; for lambada it is the single
  request. Checked against lm-eval's own `acc`: the gold index agreed on every sample checked.
- **Random baselines** in the summary: hellaswag 25, piqa 50, arc_easy 25, sciq 25, lambada 0.
- `results/summary.md` is rebuilt from every `results/*.json`: rows are task × (acc, acc_norm,
  answer BPB), with one column per model. Runs made with `--limit` are flagged as partial.

## 6. Sanity checks and results

All three checks pass.

### Randomly initialized model: holdout BPB ≈ uniform

`uv run python eval_bpb.py --random`. Uniform = log2(49152) × scored tokens / bytes.

| set | BPB | uniform |
|---|---|---|
| finemath | 5.021 | 5.005 |
| fineweb_edu | 3.204 | 3.178 |
| gutenberg | 4.023 | 3.998 |
| python_edu | 4.951 | 4.921 |
| wikipedia_en | 3.656 | 3.634 |

### Benchmarks: GPT-2 ≈ published, random init ≈ chance

GPT-2 (124M) hellaswag acc_norm is **31.1%**, the published 0-shot number. The random-init model
(exported with `export_hf.py --random`) is at chance. Its sciq acc is a bit low (19.8%), most likely
because an untrained model's summed log-likelihood favours shorter answers; acc_norm is 23.1%.

### 200-step training run: holdout BPB goes down

On the pilot mix: 200 steps × 32 sequences × 255 tokens = 1.63M training tokens, with
`eval_every_tokens=409_600` (one eval every ~50 steps) so the curve has several points. The 25M
default would only evaluate at the end of a run this short.

| tokens | fineweb_edu | finemath | gutenberg | python_edu | wikipedia_en |
|---|---|---|---|---|---|
| 0 (random init) | 3.204 | 5.021 | 4.023 | 4.951 | 3.656 |
| 416k | 2.156 | 3.052 | 2.709 | 3.198 | 2.509 |
| 824k | 2.055 | 2.878 | 2.630 | 2.954 | 2.405 |
| 1.23M | 1.991 | 2.790 | 2.537 | 2.814 | 2.342 |
| 1.63M | 1.966 | 2.746 | 2.510 | 2.754 | 2.314 |

Val loss went from 9.05 at step 10 to 6.44 at step 200; throughput was about 9k tok/s. The greedy
samples are still mostly newlines and repeated digits, as expected after 1.6M tokens. This run
did not save a checkpoint: `model.pth` was left untouched. Full log: `logs/train_200.log`.

### Benchmark results (`results/summary.md`)

`fw60_py20_fm13_gb07_s1234` is the existing `model.pth` (dated Oct 11 13:31; which run produced it
is not recorded, probably the default quick run of about 4M tokens), exported to
`exports/fw60_py20_fm13_gb07_s1234/`.

| task | metric | random | fw60_py20_fm13_gb07_s1234 | gpt2 | random_init |
|---|---|---|---|---|---|
| hellaswag | acc | 25.0 | 26.0 | 28.9 | 25.6 |
| hellaswag | acc_norm | 25.0 | 26.0 | 31.1 | 26.3 |
| hellaswag | answer BPB | | 1.895 | 0.964 | 3.410 |
| piqa | acc | 50.0 | 51.5 | 62.9 | 53.5 |
| piqa | acc_norm | 50.0 | 50.4 | 62.5 | 52.1 |
| piqa | answer BPB | | 2.201 | 1.123 | 3.585 |
| arc_easy | acc | 25.0 | 27.2 | 43.8 | 24.4 |
| arc_easy | acc_norm | 25.0 | 28.0 | 39.5 | 25.6 |
| arc_easy | answer BPB | | 1.912 | 1.248 | 2.884 |
| sciq | acc | 25.0 | 22.3 | 75.2 | 19.8 |
| sciq | acc_norm | 25.0 | 25.2 | 64.4 | 23.1 |
| sciq | answer BPB | | 2.322 | 1.405 | 2.501 |
| lambada_openai | acc | 0.0 | 0.0 | 32.6 | 0.0 |
| lambada_openai | answer BPB | | 3.075 | 0.797 | 3.215 |

The checkpoint is still at chance on accuracy, which is expected for a 40M model this early. Its
answer BPB is well below random init on every task (e.g. hellaswag 1.90 vs 3.41), which is why
answer BPB is reported: it shows progress before accuracy moves.

## Commands

Long runs: inside `tmux`, wrapped in `caffeinate -i`. If Hugging Face calls hang, prefix with
`HF_HUB_OFFLINE=1` (lm-eval needs the network the first time it downloads a task's dataset).

```bash
# Holdout sets (already built and frozen; existing sets are skipped)
uv run python make_eval_sets.py

# BPB
uv run python eval_bpb.py --check-bytes
uv run python eval_bpb.py --random
uv run python eval_bpb.py --checkpoint checkpoints/fw60_py20_fm13_gb07_s1234_124m/ckpt.pt  # holdout BPB of a checkpoint (or a weights-only .pth with --model)

# 200-step sanity run, holdout eval every ~50 steps (run id ..._124m_smoke)
HF_HUB_OFFLINE=1 uv run python -c "from config import MODEL_CONFIG_124M, TRAIN_SETTINGS; from train import main; main(MODEL_CONFIG_124M, TRAIN_SETTINGS.replace(max_steps=200, eval_every_tokens=409_600), tag='smoke')"
# The 5B run on an H100 (tmux): checkpoints every 30 min to checkpoints/<run_id>/ckpt.pt and
# s3://nano-mix/checkpoints/<run_id>/, and on Ctrl-C / SIGTERM / spot notice. Resume with the same flags:
uv run python -u train.py --model 124m --preset h100 2>&1 | tee -a logs/train_124m.log
uv run python -u train.py --model 124m --preset h100 --resume checkpoints/fw60_py20_fm13_gb07_s1234_124m/ckpt.pt 2>&1 | tee -a logs/train_124m.log
uv run python -u train.py --model 124m --preset h100 --max-steps 200 --tag smoke   # H100 smoke run

# Export (fails unless logits match within 1e-3)
uv run python export_hf.py --checkpoint checkpoints/fw60_py20_fm13_gb07_s1234_124m/ckpt.pt --out exports/fw60_py20_fm13_gb07_s1234_124m
uv run python export_hf.py --random --out exports/random_init_124m

# Benchmarks -> results/<run_name>.json, results/summary.md
uv run python run_benchmarks.py exports/fw60_py20_fm13_gb07_s1234_124m
uv run python run_benchmarks.py gpt2 HuggingFaceTB/SmolLM2-135M EleutherAI/pythia-160m
uv run python run_benchmarks.py gpt2 --limit 20     # quick smoke test, marked partial
uv run python run_benchmarks.py --summary           # only rebuild summary.md
```

## Not done yet

- **TensorBoard**: everything is logged to the terminal for now.
- **Main task set**: arc_challenge, winogrande, openbookqa and MMLU in cloze format (the right
  lm-eval task name still needs to be found). Only the pilot tasks run.
- **SmolLM2-135M and pythia-160m** have not been benchmarked yet; each is one `run_benchmarks.py`
  command.
- **Code-execution benchmarks** (HumanEval/MBPP): deliberately not set up.
- **Checkpoint provenance**: done. Checkpoints are saved under the run id
  (`checkpoints/<run_id>/ckpt.pt`) with the model, train and mix settings inside.
