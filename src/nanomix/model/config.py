"""Model architecture: ``ModelConfig`` and the named configs ("40m", "124m").

``ModelConfig`` describes anything that changes the shape of the weights, and so must match
between a training run and any later load of its checkpoint. The settings of one training run
are in train/settings.py.

A dataclass rather than a dict so a typo fails loudly at construction, and so invariants
(head_dim even for RoPE, emb_dim divisible by n_heads) are checked once, up front.
"""

import dataclasses
from dataclasses import dataclass

FFN_TYPES = ("swiglu", "gelu")
NORM_TYPES = ("rmsnorm", "layernorm")


@dataclass
class ModelConfig:
    """Architecture. Every field here is baked into the checkpoint's shapes."""

    # HuggingFaceTB/SmolLM2-135M, the tokenizer the mix was built with (CLAUDE.md).
    # 49152 = 384 * 128, so pad_vocab_to_multiple_of adds no rows.
    vocab_size: int = 49152

    # Sizes the RoPE cos/sin cache. Must equal sequence_length in configs/mix.yaml
    # (the trainer checks). 256 for the 40M Mac pilot; MODEL_CONFIG_124M uses 1024.
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
    # params). SmolLM2's 49152 is already aligned, so nothing is added. The
    # extra rows are never valid targets -- training drives their logits down
    # -- but sampling must mask them (see gpt.mask_padded_logits).
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



MODEL_CONFIG_40M = ModelConfig()  # 256 context: the Mac pilot

# ~122.7M parameters, GPT-2-small size: 37.7M in the (tied) embedding, 85.0M in
# 12 blocks of 7.08M. head_dim 64; SwiGLU hidden 2048 (swiglu_hidden_dim(768)).
# Trained on the 5B-token, 1024-context mix (configs/mix.yaml).
MODEL_CONFIG_124M = ModelConfig(context_length=1024, emb_dim=768, n_heads=12, n_layers=12)

# Named configs, chosen with --model in the trainer, eval_bpb and export_hf.
# The name goes into the run name, so 40M and 124M runs don't collide.
MODEL_CONFIGS = {"40m": MODEL_CONFIG_40M, "124m": MODEL_CONFIG_124M}


def model_label(cfg):
    """The MODEL_CONFIGS name of cfg, or a shape label for a modified config."""
    for name, named in MODEL_CONFIGS.items():
        if cfg == named:
            return name
    return f"d{cfg.emb_dim}l{cfg.n_layers}c{cfg.context_length}"
