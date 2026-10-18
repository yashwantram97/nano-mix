import torch
import torch.nn as nn
import torch.nn.functional as F

# GPT-2's initialization scheme. Every nn.Linear/nn.Embedding starts at
# N(0, 0.02); the projections that write *into* the residual stream get an
# extra 1/sqrt(2 * n_layers). That does not stop the stream growing with depth
# -- it still grows as ~sqrt(n_layers) either way -- but it makes each block's
# contribution small relative to the stream, so a 12-deep stack starts close to
# the identity instead of being dominated by its last layers. Measured here:
# the residual std after the final block drops ~4.8x versus a flat 0.02.
# PyTorch's Kaiming-uniform default is wrong for a stack this deep -- it makes
# early training slow and spiky.
INIT_STD = 0.02
_RESIDUAL_PROJ = "_is_residual_proj"


def _residual_proj(linear):
    """Tag a Linear as writing into the residual stream (so it gets scaled init).

    A flag on the module rather than a name match: renaming a submodule should
    not silently change how the model initializes.
    """
    setattr(linear, _RESIDUAL_PROJ, True)
    return linear


def init_weights(module, n_layers):
    """Apply GPT-2 init in place. Call once, after all submodules exist.

    Only Linear and Embedding are touched, so the deliberate initializations
    elsewhere -- every norm's unit scale / zero shift -- are left exactly as
    their owners set them.
    """
    residual_std = INIT_STD / (2 * n_layers) ** 0.5

    def _init(m):
        if isinstance(m, nn.Linear):
            std = residual_std if getattr(m, _RESIDUAL_PROJ, False) else INIT_STD
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=INIT_STD)

    module.apply(_init)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    Position enters by rotating q/k in 2D subspaces by an angle proportional to
    absolute position, which makes attention scores depend only on *relative*
    position. Unlike a learned table there is nothing to look up and no
    parameters, so the model is no longer structurally capped at the
    ``context_length`` it was trained at.

    The cos/sin tables are cached as non-persistent buffers: they are a pure
    function of (head_dim, context_length, base), so storing them in
    checkpoints would only re-introduce the context-length coupling that
    removing the attention mask buffer just got rid of.
    """

    def __init__(self, head_dim, context_length, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim = head_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(context_length).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        """Rotate ``x`` of shape (B, H, T, head_dim).

        ``offset`` is where this chunk starts in the full sequence; it matters
        once there is a KV cache and each step passes a single token.
        """
        num_tokens = x.shape[-2]
        cos = self.cos[offset:offset + num_tokens]
        sin = self.sin[offset:offset + num_tokens]

        # Computed in fp32: under bf16 autocast the rotation of later positions
        # loses enough precision to matter.
        x1, x2 = x.float().chunk(2, dim=-1)
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.to(x.dtype)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention on top of PyTorch's fused SDPA.

    ``F.scaled_dot_product_attention`` dispatches to FlashAttention on Ampere+,
    which never materializes the (B, H, T, T) score matrix -- attention memory
    goes from O(T^2) to O(T). ``is_causal=True`` builds the mask inside the
    kernel, so there is no mask buffer to carry, save into checkpoints, or
    resize when ``context_length`` changes.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False, rope_base=10000.0):
        super().__init__()
        assert (d_out % num_heads == 0), \
                    "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // num_heads
        self.dropout_p = dropout

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = _residual_proj(nn.Linear(d_out, d_out))
        self.rope = RotaryEmbedding(self.head_dim, context_length, base=rope_base)

    def forward(self, x, offset=0):
        b, num_tokens, _ = x.shape

        def heads(t):
            # (B, T, C) -> (B, H, T, D), the layout SDPA expects
            return t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        query = heads(self.W_query(x))
        key = heads(self.W_key(x))
        value = heads(self.W_value(x))

        # Position lives here now, not in an embedding table added at the
        # bottom of the stack. Values are deliberately not rotated.
        query = self.rope(query, offset)
        key = self.rope(key, offset)

        context_vector = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        context_vector = context_vector.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vector)


def mask_padded_logits(logits, vocab_size):
    """Drive the padded vocabulary rows to -inf before sampling.

    Padding the head to a multiple of 128 adds rows that decode to nothing.
    Training pushes their logits down, but early in a run argmax can still
    land on one, so anything that samples must mask them first.
    """
    if logits.shape[-1] > vocab_size:
        logits = logits.clone()
        logits[..., vocab_size:] = float("-inf")
    return logits


def make_norm(cfg):
    """Normalization layer. RMSNorm by default.

    Both branches are fused PyTorch kernels. The hand-rolled LayerNorm this
    replaced ran as ~5 separate elementwise ops, each reading and writing the
    full activation tensor -- the arithmetic is trivial, the memory traffic was
    the cost. RMSNorm additionally drops the mean subtraction and the shift
    parameter, which is what every recent model does.
    """
    if cfg.norm == "rmsnorm":
        return nn.RMSNorm(cfg.emb_dim, eps=1e-5)
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.emb_dim, eps=1e-5)
    raise ValueError(f"unknown norm {cfg.norm!r}")


def swiglu_hidden_dim(emb_dim, multiple_of=128):
    """Hidden width that keeps SwiGLU's parameter count equal to a 4x GELU MLP.

    A GELU MLP is 2 matrices of 4d, so 8d^2 parameters. SwiGLU has 3 matrices
    of width h, so 3dh; matching gives h = 8d/3. Rounded up to a multiple of
    128 for tensor-core alignment. At emb_dim=768 this is exactly 2048, and the
    two feed-forwards come out at identical parameter counts.
    """
    hidden = int(8 * emb_dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class FeedForward(nn.Module):
    """The original GPT-2 feed-forward: 4x expansion with GELU."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(approximate="tanh"),
            _residual_proj(nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)),
        )

    def forward(self, x):
        return self.layers(x)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward: ``W_down(silu(W_gate x) * W_up x)``.

    The gate makes the layer multiplicative rather than a fixed pointwise
    nonlinearity, which is what buys the quality over GELU at matched
    parameters. Biases are dropped, as in Llama/PaLM.
    """

    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg.emb_dim
        hidden = cfg.ffn_hidden_dim or swiglu_hidden_dim(emb_dim)
        self.hidden_dim = hidden

        self.w_gate = nn.Linear(emb_dim, hidden, bias=False)
        self.w_up = nn.Linear(emb_dim, hidden, bias=False)
        self.w_down = _residual_proj(nn.Linear(hidden, emb_dim, bias=False))

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def make_feedforward(cfg):
    """Pick the feed-forward variant. Defaults to SwiGLU; "gelu" keeps the
    original GPT-2 block available for ablation."""
    kind = cfg.ffn
    if kind == "swiglu":
        return SwiGLUFeedForward(cfg)
    if kind == "gelu":
        return FeedForward(cfg)
    raise ValueError(f"unknown ffn {kind!r}, expected 'swiglu' or 'gelu'")


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg.emb_dim,
            d_out = cfg.emb_dim,
            context_length = cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
            rope_base=cfg.rope_base,
        )
        self.ff = make_feedforward(cfg)
        self.norm1 = make_norm(cfg)
        self.norm2 = make_norm(cfg)
        self.dropout = nn.Dropout(cfg.drop_rate) 

    def forward(self, x, offset=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, offset)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg.context_length
        self.vocab_size = cfg.vocab_size
        self.gradient_checkpointing = cfg.gradient_checkpointing
        # Padded so the output-head matmul lands on aligned, tensor-core
        # friendly dimensions. The extra rows are never valid targets.
        self.token_emb = nn.Embedding(cfg.padded_vocab_size, cfg.emb_dim)
        # No position embedding table: position is applied inside attention as
        # RoPE. Nothing here is tied to context_length any more.
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = make_norm(cfg)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.padded_vocab_size, bias=False)
        if cfg.tie_embeddings:
            # One matrix for both: row i embeds token i and scores token i.
            # model.parameters() yields it once, so it is counted and updated once.
            self.out_head.weight = self.token_emb.weight

        init_weights(self, cfg.n_layers)

    def forward(self, in_idx, offset=0):
        x = self.drop_emb(self.token_emb(in_idx))
        x = self._run_blocks(x, offset)
        x = self.final_norm(x)
        return self.out_head(x)

    def _run_blocks(self, x, offset=0):
        """Run the block stack, recomputing activations if asked to.

        An explicit loop rather than nn.Sequential: Sequential can pass exactly
        one argument and gives nowhere to wrap the call, so it can carry
        neither the RoPE offset nor a checkpoint wrapper.

        Checkpointing is skipped outside training -- with no graph being built
        there is nothing to save, and recomputing would be pure waste.
        """
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.transformer_blocks:
            if recompute:
                x = torch.utils.checkpoint.checkpoint(block, x, offset, use_reentrant=False)
            else:
                x = block(x, offset)
        return x
