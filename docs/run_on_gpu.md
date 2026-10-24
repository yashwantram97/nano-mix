# Run on a GPU box: 124M / 5B on one H100

Everything from an empty GPU box to a finished run, in order. The current box is a **Vast H100
PCIe** (~$2.58/h, ~5.9 h ≈ $15); [AWS p5](#aws-p5-h100-sxm-spot) (H100 SXM, spot) is the
alternative. The steps are the same except where a section says otherwise. The smoke test runs on
the same box, right before the full run.

| run | run id | checkpoints / metrics |
|---|---|---|
| smoke test | `fw60_py20_fm13_gb07_s1234_124m_smoke` | `checkpoints/<run id>/`, `logs/<run id>.jsonl` |
| full run | `fw60_py20_fm13_gb07_s1234_124m` | same, and `s3://nano-mix/checkpoints/<run id>/` |

## 0. On the Mac, once

The eval sets are gitignored (`data/`), so they reach the box through S3. Training stops at
startup without them.

```bash
aws s3 sync data/eval s3://nano-mix/eval/
aws s3 sync data/raw_eval s3://nano-mix/raw_eval/     # only for nanomix-eval-bpb --check-bytes; small
aws s3 ls s3://nano-mix/eval/                         # 10 files: <name>.npy + <name>.json x 5
git status && git push                                # the box clones what's on GitHub
```

The token files are in `s3://nano-mix/tokenized/` (~71 GB, full) and
`s3://nano-mix/tokenized_trimmed/` (~10.1 GB, only what the mix reads; made with `nanomix-trim-mix`,
see [pipeline.md](pipeline.md)). A box outside AWS uses the trimmed copy: S3 → internet costs
$0.09/GB after the first 100 GB each month.

## 1. Rent the box

**Vast**: one H100 PCIe 80 GB, a CUDA image whose driver is **>= 580** (torch 2.13 is built for
CUDA 13.0), disk for ~10 GB of tokens + the venv + checkpoints.

Credentials on a rented box: a **read-only GitHub token for this one repo**, an **IAM user with
access to the `nano-mix` bucket only** (`aws configure` on the box), and no SSH agent forwarding.
Delete or revoke them after the run (section 9).

## 2. First login: check the GPU and driver

```bash
nvidia-smi                       # H100, "Driver Version" must be >= 580
aws sts get-caller-identity      # the nano-mix IAM user (Vast) or role (AWS)
```

If the driver is older than 580, stop and pick another image or box: torch will not see the GPU.

## 3. Tools, code, Python environment

```bash
apt-get update && apt-get install -y git tmux      # with sudo if not root
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# Private repo: paste the read-only GitHub token as the password.
git clone https://yashwantram97@github.com/yashwantram97/nano-mix.git && cd nano-mix
uv sync                          # Python 3.11 + CUDA torch from uv.lock (a few GB, ~2-3 min)
mkdir -p logs data

uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"
# expect: 2.13.0 13.0 True NVIDIA H100 PCIe (or H100 80GB HBM3 on AWS)
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

## 5. Data from S3

```bash
tmux new -s train                # everything long runs inside tmux; detach: Ctrl-b d, back: tmux attach -t train

aws configure set default.s3.max_concurrent_requests 64
aws s3 sync s3://nano-mix/tokenized_trimmed/ data/tokenized/   # Vast: ~10.1 GB (AWS: tokenized/, ~71 GB, free in-region)
aws s3 sync s3://nano-mix/eval/ data/eval/                     # 19 MB
du -sh data/tokenized/* data/eval
```

## 6. Check the mix

```bash
uv run nanomix-verify-mix 2>&1 | tee logs/verify_mix.log   # ~10 min; must end "All checks passed."
```

Don't start training if it fails. The trainer builds the same mix itself (same `configs/mix.yaml`),
so this is the check, not a required step. With the trimmed copy, `mix.py` uses its
`manifest.json` and refuses it if the mix settings changed since the trim.

## 7. Smoke test (~200 steps, ~15 min)

The H100 preset with 200 steps; warmup is lowered because it must be shorter than the run. Holdout
BPB runs once, at the end.

```bash
uv run nanomix-train --model 124m --preset h100 --max-steps 200 --set warmup_steps=20 --tag smoke \
    2>&1 | tee -ia logs/train_124m_smoke.log
```

Always `tee -ia`, never `tee -a` (CLAUDE.md). What to check in the log (and on W&B, URL printed at
the start):

| check | expect on H100 PCIe (SXM in brackets) |
|---|---|
| startup lines | `Using device: cuda (bf16)`, `MFU: 0.85 GFLOP/token, peak 756 TFLOP/s (NVIDIA H100 PCIe)` (989), `W&B: https://wandb.ai/...` |
| first step | slow (compiling, ~1-2 min) |
| `Step ... tok/s, MFU` | **~267k tok/s, MFU ~30%, ~1.96 s/step** (≥ ~349k tok/s, ~1.5 s/step) |
| `peak mem` | ~30 GB at 32 x 1024 |
| recompiles | 1-2 recompile lines at the first val loss. More during training steps: note them |
| `[eval ... tokens] bpb/...` | at the end, 5 sets, ~105 s |

**Ctrl-C and resume test.** Around step 100, press Ctrl-C once. Expect `SIGINT: saving a
checkpoint after this step`, `[checkpoint] step N -> ...`, `copied to s3://...`, then `Stopped at
step N`. Resume with exactly the same flags plus `--resume`:

```bash
uv run nanomix-train --model 124m --preset h100 --max-steps 200 --set warmup_steps=20 \
    --resume checkpoints/fw60_py20_fm13_gb07_s1234_124m_smoke/ckpt.pt 2>&1 | tee -ia logs/train_124m_smoke.log
aws s3 ls s3://nano-mix/checkpoints/fw60_py20_fm13_gb07_s1234_124m_smoke/   # ckpt.pt + the .jsonl
```

The resumed run starts at step N+1, the loss continues where it was, and the W&B chart continues.

**If MFU is well under the expected value:** stop, and bring back the log and the W&B GPU charts
(utilization, power). `nvidia-smi -q -d PERFORMANCE` shows "SW Power Cap: Active" if the card is at
its power limit.

## 8. Full run (~5.2 h of training + ~37 min of evals on PCIe)

```bash
uv run nanomix-train --model 124m --preset h100 2>&1 | tee -ia logs/train_124m.log
```

9,617 steps of 524,288 tokens (one pass over the mix), holdout BPB every 250M tokens, val loss
every 250 steps, a checkpoint every 30 min to `checkpoints/fw60_py20_fm13_gb07_s1234_124m/` and S3.

```bash
# detach: Ctrl-b d (the run keeps going); log out freely
tmux attach -t train
tail -f logs/train_124m.log
tail -n 3 logs/fw60_py20_fm13_gb07_s1234_124m.jsonl
watch -n 5 nvidia-smi                                      # second tmux window: Ctrl-b c
```

Or just the W&B page, from anywhere.

### If the box goes away

The last checkpoint is in S3 (every 30 min, and on a stop signal). On a new box, repeat sections
1-5, then:

```bash
RUN=fw60_py20_fm13_gb07_s1234_124m
aws s3 cp s3://nano-mix/checkpoints/$RUN/ckpt.pt checkpoints/$RUN/ckpt.pt
aws s3 cp s3://nano-mix/checkpoints/$RUN/$RUN.jsonl logs/$RUN.jsonl
uv run nanomix-train --model 124m --preset h100 --resume checkpoints/$RUN/ckpt.pt 2>&1 | tee -ia logs/train_124m.log
```

Same flags as the original run (add the same `--set ...` if you used one); the trainer refuses a
resume whose settings would change the training. The new box must clone a commit whose commands
match these.

## 9. After the run

```bash
# On the box: the final checkpoint and metrics are already in S3; copy the logs too.
aws s3 ls s3://nano-mix/checkpoints/fw60_py20_fm13_gb07_s1234_124m/
aws s3 sync logs s3://nano-mix/logs/
```

Destroy the box. Then revoke the GitHub token, delete the IAM user's access key, and rotate the W&B
key if you want (it was in `.env` on the box).

On the Mac (benchmarks stay on MPS):

```bash
RUN=fw60_py20_fm13_gb07_s1234_124m
aws s3 cp s3://nano-mix/checkpoints/$RUN/ckpt.pt checkpoints/$RUN/ckpt.pt   # ~1.6 GB
uv run nanomix-export-hf --checkpoint checkpoints/$RUN/ckpt.pt --out exports/$RUN
uv run nanomix-eval-bpb --checkpoint checkpoints/$RUN/ckpt.pt              # holdout BPB on the Mac
uv run nanomix-benchmarks exports/$RUN
uv run nanomix-benchmarks --summary                                        # next to gpt2, SmolLM2-135M, the 40M run
```

## AWS p5 (H100 SXM, spot)

What differs from the Vast steps above:

| setting | value |
|---|---|
| Instance type | **p5.4xlarge** (1x H100 SXM 80 GB, 16 vCPU, 256 GiB) |
| Purchase | Spot, one-time request, interruption behavior: terminate (checkpoints are in S3) |
| Availability zone | us-east-1c was cheapest (~$3.91/h); check Spot pricing history first |
| AMI | latest **AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** (driver >= 580) |
| Storage | **200 GB gp3**, throughput 500 MB/s, 6000 IOPS (71 GB of tokens + venv + checkpoints) |
| IAM instance profile | **nano-mix-ec2** (S3 read/write on `nano-mix`; no keys on the box) |
| Security group | SSH (22) from your IP only |

- Login: `ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip>`; use `sudo` for `apt-get`.
- Data: `aws s3 sync s3://nano-mix/tokenized/ data/tokenized/` (full files, ~5-10 min, free in the
  same region).
- Spot: the trainer polls EC2 instance metadata and checkpoints on the 2-minute interruption notice
  (`watch_spot`). Off EC2 that check ends on its own.
- Expected: MFU >= 30% is >= ~349k tok/s at 989 TFLOP/s; ~4-5 h for the full run.
- After the run: terminate the instance in the EC2 console.
