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
DELTA_BACKENDS = ("auto", "fla", "chunk", "recurrent")
DELTA_CHUNK_SIZES = (16, 32, 64)
PRECISIONS = ("bf16", "fp32")
LR_SCHEDULES = ("cosine", "constant")


@dataclass
class ModelConfig:
    """Architecture. Every field here is baked into the checkpoint's shapes."""

    # HuggingFaceTB/SmolLM2-135M, the tokenizer the mix was built with (CLAUDE.md).
    # 49152 = 384 * 128, so pad_vocab_to_multiple_of adds no rows.
    vocab_size: int = 49152

    # Sizes the RoPE cos/sin cache. Must equal sequence_length in mix.yaml: every
    # mix instance is 256 tokens (train.py checks). 256, not 2048, to keep
    # attention cheap enough for local training. With RoPE there is no
    # position table, so this is not baked into the weights.
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

    # Gated delta rule backend; "auto" resolves to the flash-linear-attention
    # Triton kernel on CUDA and the pure-PyTorch chunkwise form elsewhere.
    delta_backend: str = "auto"
    delta_chunk_size: int = 64

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
        if self.delta_backend not in DELTA_BACKENDS:
            raise ValueError(
                f"delta_backend must be one of {DELTA_BACKENDS}, got {self.delta_backend!r}")
        if self.delta_chunk_size not in DELTA_CHUNK_SIZES:
            raise ValueError(
                f"delta_chunk_size must be one of {DELTA_CHUNK_SIZES}, "
                f"got {self.delta_chunk_size}")

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
    """One run. None of this affects checkpoint shapes."""

    learning_rate: float = 5e-4
    # Total *optimizer* steps -- the run length. One pass over the mix: 1,953,792
    # instances - val_instances = 1,945,600 = 950 steps of 2048 sequences.
    # train.py refuses to run more steps than the mix has (no repetition).
    max_steps: int = 500        # quick local test run; 950 = one full pass
    eval_freq: int = 10         # Evaluate every N steps
    eval_iter: int = 20         # Batches averaged per evaluation
    # Holdout BPB on the frozen sets in data/eval/ (eval_bpb.py), plus greedy samples from fixed
    # prompts, every this many training tokens and at the end of the run.
    eval_every_tokens: int = 25_000_000
    sample_freq: int = 250      # Generate a text sample every N steps
    weight_decay: float = 0.1

    # Micro-batch: how many sequences fit in memory at once.
    batch_size: int = 32        # quick local test; 64 for a real run

    # Micro-batches accumulated before each optimizer step. The effective
    # batch is batch_size * grad_accum_steps * context_length tokens. Costs no
    # extra compute -- the same total FLOPs, just fewer optimizer updates --
    # and is the only way to reach a sane batch size when memory caps
    # batch_size. GPT-2 small was pretrained near 500k tokens per step; at
    # batch_size=2 and context_length=256 an accum of 1 gives 512, which is
    # far too noisy for a stable run. 64 x 32 = 2048 sequences x 256 = 524,288
    # tokens, the global_batch_size the mix was sized for in mix.yaml. On a
    # bigger GPU, raise batch_size and lower this, keeping the product 2048.
    # Set to 1 for a quick local test (32 x 256 = 8,192 tokens/step); use
    # 32 with batch_size=64 for a real run.
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
    warmup_steps: int = 5       # must be < max_steps; use 100 with max_steps=950
    min_lr_ratio: float = 0.1

    # Autocast dtype for forward and loss. bf16 has fp32's exponent range, so
    # unlike fp16 it needs no GradScaler. Falls back to fp32 on CUDA hardware
    # older than Ampere, which has no bf16 support.
    precision: str = "bf16"

    # torch.compile the model: fuses the many small ops of a 40M model into
    # fewer kernels, typically 1.3-2x faster, after ~1 min of compiling at the
    # start. CUDA only; MPS/CPU support is unreliable, so it is skipped there.
    compile: bool = True

    # DataLoader worker processes. 0 loads batches on the training process,
    # which starves the GPU between steps.
    num_workers: int = 4

    # Training data: the OLMo-core mix described by this file (mix.py builds it;
    # verify_mix.py checks it).
    mix_config: str = "mix.yaml"

    # Mix instances held out for validation, chosen with mix.yaml's seed and
    # never trained on. 8192 x 256 = 2.1M tokens.
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

    def tokens_per_optimizer_step(self, context_length):
        return self.batch_size * self.grad_accum_steps * context_length

    def replace(self, **changes):
        """A modified copy, re-validated."""
        return dataclasses.replace(self, **changes)


MODEL_CONFIG = ModelConfig()
TRAIN_SETTINGS = TrainSettings()
