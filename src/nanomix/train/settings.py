"""Settings of one training run (``TrainSettings``) and the named presets ("mac", "h100").

None of this changes the shape of the weights (that is model/config.py). A resume checks that
the settings that change the training are the same (checkpoint.py RESUME_SAFE lists the rest).
"""

import dataclasses
from dataclasses import dataclass

from nanomix.paths import MIX_CONFIG

PRECISIONS = ("bf16", "fp32")
LR_SCHEDULES = ("cosine", "constant")


@dataclass
class TrainSettings:
    """One run. None of this affects checkpoint shapes.

    The defaults are a quick Mac (MPS) test (MAC_SETTINGS); H100_SETTINGS below is the 124M / 5B run.
    """

    learning_rate: float = 5e-4
    # Total *optimizer* steps -- the run length. One pass over the 5B mix: 4,932,096
    # instances - val_instances 8,192 = 4,923,904 = 9,617 steps of 512 sequences.
    # The trainer refuses to run more steps than the mix has (no repetition).
    max_steps: int = 500        # quick local test run
    # A train-loss line (mean over the steps since the last line, tok/s, MFU) every log_freq
    # steps. It costs one GPU sync, nothing else.
    log_freq: int = 10
    # Validation loss every eval_freq steps, on eval_iter fixed val batches loaded once at the
    # start (the same batches every time, so the curve isn't noise from changing batches).
    eval_freq: int = 50
    eval_iter: int = 20
    # Holdout BPB on the frozen sets in data/eval/ (eval/bpb.py), every this many training tokens
    # and at the end of the run.
    eval_every_tokens: int = 25_000_000
    # Greedy samples from the fixed prompts (eval/sample.py) every N steps and at the end.
    sample_freq: int = 250

    # AdamW. Decay applies to the weight matrices only: norms, biases and the (tied)
    # embedding get none. beta2 0.95 instead of PyTorch's 0.999, as GPT-2/3 and most LLM
    # runs use: the second moment adapts faster, which keeps large-batch runs stable.
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)

    # Micro-batch: how many sequences fit in memory at once. 8 x 1024 keeps the 124M model's
    # logits (49,152 per token) small enough for a Mac.
    batch_size: int = 8

    # Micro-batches accumulated before each optimizer step. The effective
    # batch is batch_size * grad_accum_steps * context_length tokens. Costs no
    # extra compute -- the same total FLOPs, just fewer optimizer updates --
    # and is the only way to reach a sane batch size when memory caps
    # batch_size. GPT-2 small was pretrained near 500k tokens per step:
    # 512 sequences x 1024 = 524,288 tokens, the global_batch_size the mix was
    # sized for in mix.yaml (H100_SETTINGS: 32 x 16). 1 for a quick local test
    # (8 x 1024 = 8,192 tokens/step).
    grad_accum_steps: int = 1

    # Clip the global gradient norm before each optimizer step. Set to 0 or
    # None to disable. Applied *after* accumulation, so it clips the true
    # full-batch gradient rather than each micro-batch separately.
    grad_clip: float = 1.0

    # Learning rate schedule. Linear warmup from 0 over warmup_steps, then
    # "cosine" decay to min_lr_ratio * learning_rate at max_steps, or
    # "constant" to hold the peak. Warmup matters most in the first few
    # hundred steps, when Adam's second-moment estimates are still noisy and a
    # full-size step can blow the model out.
    lr_schedule: str = "cosine"
    warmup_steps: int = 5       # must be < max_steps
    min_lr_ratio: float = 0.1

    # Autocast dtype for forward and loss. bf16 has fp32's exponent range, so
    # unlike fp16 it needs no GradScaler. Falls back to fp32 on CUDA hardware
    # older than Ampere, which has no bf16 support.
    precision: str = "bf16"

    # torch.compile the model: fuses the many small ops into fewer kernels,
    # typically 1.3-2x faster, after ~1 min of compiling at the start. CUDA
    # only; MPS/CPU support is unreliable, so it is skipped there.
    compile: bool = True

    # Training DataLoader worker processes, kept alive for the whole run. 0 loads
    # batches on the training process, which starves the GPU between steps.
    num_workers: int = 4

    # Dense bf16 peak of the GPU in TFLOP/s, for MFU. None looks the GPU up by name
    # (train/metrics.py PEAK_TFLOPS); if it isn't there, MFU isn't printed.
    peak_tflops: float | None = None

    # Metrics (train / val / BPB records) are appended to <metrics_dir>/<run_id>.jsonl.
    metrics_dir: str = "logs"

    # Checkpoints (train/checkpoint.py): <checkpoint_dir>/<run_id>/ckpt.pt every checkpoint_minutes
    # (None: only on a stop signal and at the end), the previous one kept as ckpt_prev.pt. With
    # checkpoint_s3 set, each one (and the metrics file) is also copied to <checkpoint_s3>/<run_id>/.
    # watch_spot polls EC2 instance metadata for a spot interruption notice and checkpoints on it.
    checkpoint_minutes: float | None = 30
    checkpoint_dir: str = "checkpoints"
    checkpoint_s3: str | None = None
    watch_spot: bool = True

    # Weights & Biases: with wandb_project set, every train / val / BPB record also goes to W&B
    # (plus stdout and GPU stats), and alerts fire on a non-finite loss or a large tok/s drop.
    # The key comes from WANDB_API_KEY (environment or .env). None: no W&B, JSONL only.
    wandb_project: str | None = None
    wandb_entity: str | None = None     # None: your W&B user

    # Training data: the OLMo-core mix described by this file (mix.py builds it;
    # verify_mix.py checks it).
    mix_config: str = str(MIX_CONFIG)

    # Mix instances held out for validation, chosen with the mix's seed and
    # never trained on. 8192 x 1024 = 8.4M tokens.
    val_instances: int = 8192

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} must be < max_steps {self.max_steps}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(
                f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.log_freq < 1 or self.eval_freq < 1:
            raise ValueError(
                f"log_freq and eval_freq must be >= 1, got {self.log_freq}, {self.eval_freq}")

    def tokens_per_optimizer_step(self, context_length):
        return self.batch_size * self.grad_accum_steps * context_length

    def replace(self, **changes):
        """A modified copy, re-validated."""
        return dataclasses.replace(self, **changes)


MAC_SETTINGS = TrainSettings()  # quick Mac test

# The 124M / 5B-token run on one H100 80GB: one pass over the whole mix.
# Starting point from GPT-2 small; batch_size 64 x 8 also gives 512 sequences if it fits
# (check peak memory in the smoke run).
H100_SETTINGS = MAC_SETTINGS.replace(
    learning_rate=6e-4,
    max_steps=9_617,            # 4,923,904 train instances / 512 per step
    warmup_steps=300,
    batch_size=64,
    grad_accum_steps=8,        # 32 x 16 x 1024 = 524,288 tokens/step
    log_freq=10,
    eval_freq=250,
    eval_every_tokens=250_000_000,  # 20 holdout evals over the run
    sample_freq=1000,
    num_workers=8,              # p5.4xlarge has 16 vCPUs
    checkpoint_s3="s3://nano-mix/checkpoints",
    wandb_project="nano-mix",
)

# Chosen with --preset.
TRAIN_PRESETS = {"mac": MAC_SETTINGS, "h100": H100_SETTINGS}
