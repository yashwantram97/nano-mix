# Training

How `nanomix-train` (`src/nanomix/train/cli.py`) trains the GPT on the mix. Fixed choices and their
reasons are in [decisions.md](decisions.md); commands for a GPU box in [run_on_gpu.md](run_on_gpu.md).

## Files

| file | what |
|---|---|
| `model/config.py` | `ModelConfig` (anything that changes weight shapes) and `MODEL_CONFIGS`: "40m" (Mac pilot, 256 context), "124m" (1024 context) |
| `model/gpt.py` | `GPTModel`: RoPE, RMSNorm pre-norm blocks, SwiGLU, SDPA attention, tied embedding, GPT-2 init |
| `train/settings.py` | `TrainSettings` (one run) and `TRAIN_PRESETS`: "mac" (quick test), "h100" (the 5B run) |
| `train/cli.py` | `--model` / `--preset` / `--set` / `--resume`; builds model, optimizer, data, logging, checkpoints |
| `train/loop.py` | `train`: optimizer steps, val loss, holdout BPB, samples, checkpoints |
| `train/optim.py` | `param_groups` (decay on matrices only), `lr_at_step` (warmup + cosine) |
| `train/precision.py` | device choice, bf16 autocast |
| `train/metrics.py` | JSONL metrics, W&B and alerts, MFU, the loss plot |
| `train/checkpoint.py` | `Checkpointer`, `StopRequest`, `check_resume` |
| `data/loader.py` | the mix as batches: val split, seeded order, skip on resume |

## The 124M run

| | value |
|---|---|
| model | 12 layers, width 768, 12 heads (head 64), SwiGLU 2048, tied 49,152 embedding: 122.7M parameters |
| tokens per step | 32 × 16 micro-batches × 1024 = 524,288 |
| steps | 4,923,904 training instances / 512 = 9,617 (5.04B tokens, one pass) |
| LR | warmup 300 steps to 6e-4, cosine to 6e-5 |
| AdamW | betas (0.9, 0.95), weight decay 0.1 on matrices, fused |
| precision | bf16 autocast for forward + loss; fp32 weights, gradients and optimizer state |
| clipping | total gradient norm 1.0, once per step, after accumulation |

## Data

`create_mix_dataloaders` builds the mix with the same code as `nanomix-mix`, so training reads the
instances `nanomix-verify-mix` checked. A `torch.randperm` with the mix seed orders the instances;
the first `val_instances` (8,192) are the validation set, the rest are read once, in that order.
Each instance gives (input, target): the target is the input shifted left by one, and the last
position is `IGNORE_INDEX` (-100), so 1,023 of 1,024 positions are scored.

## The loop, every step

1. Set the LR for this optimizer step.
2. `grad_accum_steps` micro-batches: forward + loss under autocast, `(loss / accum).backward()`.
3. Clip the accumulated gradient, `optimizer.step()`.

Losses and grad norms stay on the GPU; `.item()` (a GPU→CPU sync) is called only for a log line.
`train_model` (compiled) runs the steps; `model` (the same weights, uncompiled) does sampling and
holdout BPB, so their other shapes don't recompile the training graph.

## Output

| when (h100 preset) | what |
|---|---|
| every `log_freq` (10) steps | step line: mean loss, lr, grad norm before clipping, clip fraction, tok/s, s/step, MFU, peak memory |
| every `eval_freq` (250) steps | val line: train loss (mean since the last val line), val loss on `eval_iter` (20) fixed val batches, param norm |
| every `sample_freq` (1,000) steps and at the end | greedy samples from the prose, code and math prompts (`eval/sample.py`) |
| every `eval_every_tokens` (250M) and at the end | holdout BPB on the 5 frozen sets (`eval/bpb.py`), fp32 |
| every `checkpoint_minutes` (30), on a stop, and at the end | checkpoint |
| at the end (or a stop) | `logs/<run_id>_loss.pdf` |

Throughput covers training only: time on val loss, samples, evals and checkpoints is excluded.
MFU = tok/s × FLOPs per token (6N + attention) ÷ the GPU's dense bf16 peak (`metrics.py`
`PEAK_TFLOPS`: 756 TFLOP/s for the H100 PCIe, 989 for SXM).

Every record also goes to `logs/<run_id>.jsonl` (after a `run` record with the full config), and,
with `wandb_project` set (h100 preset: `nano-mix`, key `WANDB_API_KEY` from `.env`), to W&B. W&B
alerts fire on a non-finite loss (the run then stops **without** saving, to keep the last good
checkpoint) and on tok/s below half the best so far.

## Checkpoints, stops and resume

A checkpoint (`checkpoints/<run_id>/ckpt.pt`, ~1.6 GB) holds the weights, AdamW state, step, tokens
seen, RNG states, the loss history, and the model / train / mix settings. Tensors are on CPU and all
values are plain types, so `torch.load(..., weights_only=True)` reads it anywhere, and modules can be
moved without breaking old checkpoints. The previous one is kept as `ckpt_prev.pt`; with
`checkpoint_s3` set, each one and the metrics file are copied to S3 in the background.

`StopRequest` catches Ctrl-C, SIGTERM, SIGHUP and an EC2 spot notice: the loop finishes the step,
saves, waits for the S3 copy and exits. A second Ctrl-C quits without saving. DataLoader workers
ignore these signals, so the main process can save.

`--resume checkpoints/<run_id>/ckpt.pt` with the same `--model`, `--preset` and `--set` continues the
run: the loader skips `step × grad_accum_steps × batch_size` instances, so the next batch is exactly
the one that would have come. `check_resume` refuses a different model, a changed train setting
(except those in `RESUME_SAFE`: logging, saving and speed) or a changed mix (`MIX_KEYS`). On a GPU the
resume isn't bit-exact (see decisions.md); the data and schedule are.
