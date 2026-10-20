"""Training configuration.

Dataclasses rather than dicts so a typo fails loudly at construction instead
of surfacing as a KeyError deep inside a forward pass twenty minutes into a
run, and so invariants (head_dim even for RoPE, emb_dim divisible by n_heads)
are checked once, up front.

``ModelConfig`` describes the architecture -- anything that changes the shape
of the weights, and so must match between a training run and any later load of
its checkpoint. ``TrainSettings`` describes a particular run and can change
freely between runs.
"""

import dataclasses
from dataclasses import dataclass

FFN_TYPES = ("swiglu", "gelu")
NORM_TYPES = ("rmsnorm", "layernorm")
PRECISIONS = ("bf16", "fp32")
LR_SCHEDULES = ("cosine", "constant")


@dataclass
class ModelConfig:
    """Architecture. Every field here is baked into the checkpoint's shapes."""

    # HuggingFaceTB/SmolLM2-135M, the tokenizer the mix was built with (CLAUDE.md).
    # 49152 = 384 * 128, so pad_vocab_to_multiple_of adds no rows.
    vocab_size: int = 49152

    # Sizes the RoPE cos/sin cache. Must equal sequence_length in mix.yaml
    # (train.py checks). 256 for the 40M Mac pilot; MODEL_CONFIG_124M uses 1024.
    # With RoPE there is no position table, so this is not baked into the
    # weights, but a model is only reliable up to the length it was trained at.
    context_length: int = 256

    # ~40M parameters: 18.9M in the (tied) embedding, 21.2M in 12 blocks of
    # 1.77M. Deep and narrow, as MobileLLM found works best at this size;
    # head_dim 64.
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 12

    # Share the token embedding with the output head (as SmolLM2 does). With a
    # 49k vocabulary each copy is 18.9M parameters at emb_dim 384, so untied
    # they would be ~half of a model this size.
    tie_embeddings: bool = True

    # 0.0, not 0.1: this is a single pass over the mix, so the model never
    # sees a token twice and there is nothing to overfit. Dropout here only adds
    # gradient noise and costs throughput.
    drop_rate: float = 0.0
    qkv_bias: bool = False

    # Feed-forward variant. ffn_hidden_dim overrides the 8/3 * emb_dim default,
    # which is chosen so SwiGLU matches the GELU MLP's parameter count.
    ffn: str = "swiglu"
    ffn_hidden_dim: int | None = None

    # Normalization. RMSNorm drops the mean-subtraction and the shift parameter;
    # both options map to fused PyTorch kernels.
    norm: str = "rmsnorm"

    # RoPE rotation base. 10000 is standard; raising it is the usual first move
    # when extending to longer contexts.
    rope_base: float = 10000.0

    # Recompute each block's activations during backward instead of storing
    # them: roughly 33% more compute for a large drop in activation memory,
    # which is what lets batch size go up. Gradients are unchanged.
    gradient_checkpointing: bool = False

    # Round the embedding and output head up to a multiple of this. 50257 is
    # odd, so the head matmul cannot use vectorized loads or hit the fast
    # tensor-core kernels; 50304 = 393 * 128 costs 47 unused rows (0.036M
    # params). SmolLM2's 49152 is already aligned, so nothing is added. The extra rows are never valid targets -- training drives their
    # logits down -- but sampling must mask them (see mask_padded_logits).
    pad_vocab_to_multiple_of: int = 128

    def __post_init__(self):
        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim {self.emb_dim} not divisible by n_heads {self.n_heads}")
        if self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")
        if self.ffn not in FFN_TYPES:
            raise ValueError(f"ffn must be one of {FFN_TYPES}, got {self.ffn!r}")
        if self.norm not in NORM_TYPES:
            raise ValueError(f"norm must be one of {NORM_TYPES}, got {self.norm!r}")

    @property
    def head_dim(self):
        return self.emb_dim // self.n_heads

    @property
    def padded_vocab_size(self):
        """Vocabulary the weights actually use. Always >= vocab_size."""
        multiple = self.pad_vocab_to_multiple_of
        if multiple <= 1:
            return self.vocab_size
        return multiple * ((self.vocab_size + multiple - 1) // multiple)

    def replace(self, **changes):
        """A modified copy, re-validated. ``cfg.replace(n_layers=2)``."""
        return dataclasses.replace(self, **changes)


@dataclass
class TrainSettings:
    """One run. None of this affects checkpoint shapes.

    The defaults are a quick Mac (MPS) test; H100_SETTINGS below is the 124M / 5B run.
    """

    learning_rate: float = 5e-4
    # Total *optimizer* steps -- the run length. One pass over the 5B mix: 4,932,096
    # instances - val_instances 8,192 = 4,923,904 = 9,617 steps of 512 sequences.
    # train.py refuses to run more steps than the mix has (no repetition).
    max_steps: int = 500        # quick local test run
    # A train-loss line (mean over the steps since the last line, tok/s, MFU) every log_freq
    # steps. It costs one GPU sync, nothing else.
    log_freq: int = 10
    # Validation loss every eval_freq steps, on eval_iter fixed val batches loaded once at the
    # start (the same batches every time, so the curve isn't noise from changing batches).
    eval_freq: int = 50
    eval_iter: int = 20
    # Holdout BPB on the frozen sets in data/eval/ (eval_bpb.py), plus greedy samples from fixed
    # prompts, every this many training tokens and at the end of the run.
    eval_every_tokens: int = 25_000_000
    sample_freq: int = 250      # Generate a text sample every N steps

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
    # (train.py PEAK_TFLOPS); if it isn't there, MFU isn't printed.
    peak_tflops: float | None = None

    # Metrics (train / val / BPB records) are appended to <metrics_dir>/<run_id>.jsonl.
    metrics_dir: str = "logs"

    # Checkpoints (checkpointing.py): <checkpoint_dir>/<run_id>/ckpt.pt every checkpoint_minutes
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
    mix_config: str = "mix.yaml"

    # Mix instances held out for validation, chosen with mix.yaml's seed and
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


MODEL_CONFIG = ModelConfig()  # 40M, 256 context: the Mac pilot

# ~122.7M parameters, GPT-2-small size: 37.7M in the (tied) embedding, 85.0M in
# 12 blocks of 7.08M. head_dim 64; SwiGLU hidden 2048 (swiglu_hidden_dim(768)).
# Trained on the 5B-token, 1024-context mix (mix.yaml).
MODEL_CONFIG_124M = ModelConfig(context_length=1024, emb_dim=768, n_heads=12, n_layers=12)

# Named configs, chosen with --model in train.py, eval_bpb.py and export_hf.py.
# The name goes into the run name, so 40M and 124M runs don't collide.
MODEL_CONFIGS = {"40m": MODEL_CONFIG, "124m": MODEL_CONFIG_124M}


def model_label(cfg):
    """The MODEL_CONFIGS name of cfg, or a shape label for a modified config."""
    for name, named in MODEL_CONFIGS.items():
        if cfg == named:
            return name
    return f"d{cfg.emb_dim}l{cfg.n_layers}c{cfg.context_length}"
TRAIN_SETTINGS = TrainSettings()  # quick Mac test

# The 124M / 5B-token run on one H100 80GB: one pass over the whole mix.
# Starting point from GPT-2 small; batch_size 64 x 8 also gives 512 sequences if it fits
# (check peak memory in the smoke run).
H100_SETTINGS = TRAIN_SETTINGS.replace(
    learning_rate=6e-4,
    max_steps=9_617,            # 4,923,904 train instances / 512 per step
    warmup_steps=300,
    batch_size=32,
    grad_accum_steps=16,        # 32 x 16 x 1024 = 524,288 tokens/step
    log_freq=10,
    eval_freq=250,
    eval_every_tokens=250_000_000,  # 20 holdout evals over the run
    sample_freq=1000,
    num_workers=8,              # p5.4xlarge has 16 vCPUs
    checkpoint_s3="s3://nano-mix/checkpoints",
    wandb_project="nano-mix",
)

# Chosen with train.py --preset.
TRAIN_PRESETS = {"mac": TRAIN_SETTINGS, "h100": H100_SETTINGS}
