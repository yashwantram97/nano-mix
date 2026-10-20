# Startup commands: 124M / 5B run on one H100 (AWS p5.4xlarge)

Everything from an empty GPU box to a finished run, in order. The smoke test runs on the same
box, right before the full run. Run ids used below:

| run | run id | checkpoints / metrics |
|---|---|---|
| smoke test | `fw60_py20_fm13_gb07_s1234_124m_smoke` | `checkpoints/<run id>/`, `logs/<run id>.jsonl` |
| full run | `fw60_py20_fm13_gb07_s1234_124m` | same, and `s3://nano-mix/checkpoints/<run id>/` |

## 0. On the Mac, once

The eval sets are gitignored (`data/`), so they reach the box through S3. Training stops at
startup without them.

```bash
aws s3 sync data/eval s3://nano-mix/eval/
aws s3 sync data/raw_eval s3://nano-mix/raw_eval/     # only for eval_bpb --check-bytes; small
aws s3 ls s3://nano-mix/eval/                         # 10 files: <name>.npy + <name>.json x 5
git status && git push                                # the box clones what's on GitHub
```

## 1. Launch the instance (AWS console, us-east-1)

| setting | value |
|---|---|
| Instance type | **p5.4xlarge** (1x H100 SXM 80 GB, 16 vCPU, 256 GiB) |
| Purchase | Spot, one-time request, interruption behavior: terminate (checkpoints are in S3) |
| Availability zone | us-east-1c was cheapest (~$3.91/h); check Spot pricing history first |
| AMI | latest **AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**. torch 2.13 is built for CUDA 13.0 and needs **driver >= 580** |
| Storage | **200 GB gp3**, throughput 500 MB/s, 6000 IOPS (71 GB of tokens + venv + checkpoints; faster S3 sync) |
| IAM instance profile | **nano-mix-ec2** (S3 read/write on `nano-mix`, used for the data and checkpoint copies) |
| Security group | SSH (22) from your IP only |
| Key pair | yours |

Budget check: AWS Budgets alert is set. Smoke test ~30-40 min, full run ~4-5 h at 30% MFU.

## 2. First login: check the GPU and driver

```bash
ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip>

nvidia-smi                       # H100 80GB HBM3, "Driver Version" must be >= 580
aws sts get-caller-identity      # shows the nano-mix-ec2 role (no keys needed)
```

If the driver is older than 580, stop here and pick a newer AMI: torch will not see the GPU.

## 3. Tools, code, Python environment

```bash
sudo apt-get update && sudo apt-get install -y git tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# Private repo: paste a GitHub personal access token as the password. Revoke it after the run.
git clone https://yashwantram97@github.com/yashwantram97/nano-mix.git && cd nano-mix
uv sync                          # Python 3.11 + CUDA torch from uv.lock (a few GB, ~2-3 min)
mkdir -p logs data

uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"
# expect: 2.13.0 13.0 True NVIDIA H100 80GB HBM3
uv run python - <<'EOF'
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 12, 1024, 64, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # errors if FlashAttention can't run
    F.scaled_dot_product_attention(q, q, q, is_causal=True)
print("FlashAttention OK")
EOF
```

## 4. W&B key

`.env` is gitignored, so it doesn't come with the clone. Create it on the box:

```bash
echo "WANDB_API_KEY=<your key>" > .env && chmod 600 .env
```

In W&B, User settings -> Alerts: turn on email (or Slack) for run finished / crashed and scripted
alerts, or the NaN and slowdown alerts won't reach you.

## 5. Data from S3 (same region: free)

```bash
tmux new -s train                # everything long runs inside tmux; detach: Ctrl-b d, back: tmux attach -t train

aws configure set default.s3.max_concurrent_requests 64
aws s3 sync s3://nano-mix/tokenized/ data/tokenized/    # ~71 GB, ~5-10 min
aws s3 sync s3://nano-mix/eval/ data/eval/              # 19 MB
du -sh data/tokenized/* data/eval                        # fineweb_edu / python_edu / finemath / gutenberg + eval
```

## 6. Build and check the mix

```bash
uv run python -u mix.py 2>&1 | tee logs/mix.log                 # coverage > 1.0 for all 4 sources; ~4,932,096 instances
uv run python -u verify_mix.py 2>&1 | tee logs/verify_mix.log   # ~10 min; must end "All checks passed."
```

Don't start training if `verify_mix.py` fails. `train.py` builds the same mix itself (same
`mix.yaml`), so this is the check, not a required step for training.

### Optional: a trimmed copy for boxes outside AWS (~10 GB instead of 71 GB)

The mix reads only the start of each token file. `trim_mix.py` copies just those tokens, plus a
`manifest.json` with the per-file allocation, then rebuilds the mix from the copy and checks the
instances are identical. Run it here, where the full files are:

```bash
uv run python -u trim_mix.py 2>&1 | tee logs/trim_mix.log   # writes data/tokenized_trimmed/; must end "PASS"
aws s3 sync data/tokenized_trimmed/ s3://nano-mix/tokenized_trimmed/   # or to R2 (--endpoint-url ...)
```

On the other box, sync it to `data/tokenized/` (mix.yaml's `tokenized_dir`) and run `verify_mix.py`.
`mix.py` sees `manifest.json` and uses the recorded allocation; it refuses the copy if `mix.yaml`
changed since the trim, so trim again after any mix change. Resuming works across the two: the
instances and their order are the same.

## 7. Smoke test (~200 steps, ~15 min)

The H100 preset with 200 steps; warmup is lowered because it must be shorter than the run.
Holdout BPB runs once, at the end.

```bash
uv run python -u train.py --model 124m --preset h100 --max-steps 200 --set warmup_steps=20 --tag smoke \
    2>&1 | tee -a logs/train_124m_smoke.log
```

What to check in the log (and on W&B, URL printed at the start):

| check | expect |
|---|---|
| startup lines | `Using device: cuda (bf16)`, `MFU: 0.85 GFLOP/token, peak 989 TFLOP/s (NVIDIA H100 80GB HBM3)`, `W&B: https://wandb.ai/...` |
| first step | slow (compiling, ~1-2 min) |
| `Step ... tok/s, MFU` | **MFU >= 30% (>= ~349k tok/s)**, ~1.5 s/step |
| `peak mem` | well under 80 GB at 32 x 1024 (decides whether to try 64 x 8) |
| recompiles | 1-2 recompile lines at the first val loss (step 200 here; every 250 steps in the full run). More during training steps: note them |
| `[eval ... tokens] bpb/...` | at the end, 5 sets, no error |

**Ctrl-C and resume test.** Around step 100, press Ctrl-C once. Expect `SIGINT: saving a
checkpoint after this step`, `[checkpoint] step N -> ...`, `copied to s3://...`, then `Stopped at
step N`. Resume with exactly the same flags plus `--resume`:

```bash
uv run python -u train.py --model 124m --preset h100 --max-steps 200 --set warmup_steps=20 \
    --resume checkpoints/fw60_py20_fm13_gb07_s1234_124m_smoke/ckpt.pt 2>&1 | tee -a logs/train_124m_smoke.log
aws s3 ls s3://nano-mix/checkpoints/fw60_py20_fm13_gb07_s1234_124m_smoke/   # ckpt.pt + the .jsonl
```

The resumed run starts at step N+1, the loss continues where it was, and the W&B chart continues.

**Optional: does 64 x 8 fit (and is it faster)?** Only if peak memory at 32 x 1024 was well
under ~40 GB:

```bash
uv run python -u train.py --model 124m --preset h100 --max-steps 40 --set warmup_steps=5 batch_size=64 grad_accum_steps=8 \
    --tag smoke64 2>&1 | tee -a logs/train_124m_smoke64.log      # Ctrl-C after ~20 steps
```

Keep 64 x 8 only if it doesn't run out of memory and tok/s is clearly higher. If you keep it, the
full run and every resume of it need `--set batch_size=64 grad_accum_steps=8`.

**If MFU is under 30%:** stop here, and bring back the log (`logs/train_124m_smoke.log`) and the
W&B GPU charts (utilization, power). Don't start the full run yet.

## 8. Full run (~4-5 h)

```bash
uv run python -u train.py --model 124m --preset h100 2>&1 | tee -a logs/train_124m.log
```

9,617 steps of 524,288 tokens (one pass over the mix), holdout BPB every 250M tokens, val loss
every 250 steps, a checkpoint every 30 min to `checkpoints/fw60_py20_fm13_gb07_s1234_124m/` and S3.

Watching it:

```bash
# detach: Ctrl-b d (the run keeps going); log out freely
tmux attach -t train                                       # back to the run
tail -f logs/train_124m.log
tail -n 3 logs/fw60_py20_fm13_gb07_s1234_124m.jsonl
watch -n 5 nvidia-smi                                      # second tmux window: Ctrl-b c
```

Or just the W&B page, from anywhere.

## 9. If the spot instance is interrupted

The run saves a checkpoint on the 2-minute spot notice and copies it to S3 (otherwise the last
30-min checkpoint in S3 is used). On a new box, repeat sections 1-5 (section 6 isn't needed:
`train.py` builds the mix), then:

```bash
RUN=fw60_py20_fm13_gb07_s1234_124m
aws s3 cp s3://nano-mix/checkpoints/$RUN/ckpt.pt checkpoints/$RUN/ckpt.pt
aws s3 cp s3://nano-mix/checkpoints/$RUN/$RUN.jsonl logs/$RUN.jsonl
uv run python -u train.py --model 124m --preset h100 --resume checkpoints/$RUN/ckpt.pt 2>&1 | tee -a logs/train_124m.log
```

Same flags as the original run (add the same `--set ...` if you used one); `train.py` refuses a
resume whose settings would change the training.

## 10. After the run

```bash
# On the box: the final checkpoint and metrics are already in S3; copy the logs too, then terminate.
aws s3 ls s3://nano-mix/checkpoints/fw60_py20_fm13_gb07_s1234_124m/
aws s3 sync logs s3://nano-mix/logs/
```

Terminate the instance (EC2 console). Then revoke the GitHub token used for the clone, and rotate
the W&B key if you want (it was in `.env` on the box, which is deleted with it).

On the Mac (benchmarks stay on MPS):

```bash
RUN=fw60_py20_fm13_gb07_s1234_124m
aws s3 cp s3://nano-mix/checkpoints/$RUN/ckpt.pt checkpoints/$RUN/ckpt.pt   # ~1.5 GB, free (under 100 GB/month egress)
uv run python export_hf.py --checkpoint checkpoints/$RUN/ckpt.pt --out exports/$RUN
uv run python eval_bpb.py --checkpoint checkpoints/$RUN/ckpt.pt            # holdout BPB on the Mac
uv run python run_benchmarks.py exports/$RUN
uv run python run_benchmarks.py --summary                                  # next to gpt2, SmolLM2-135M, the 40M run
```
