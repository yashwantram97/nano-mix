"""Convert a GPTModel checkpoint (train.py's model.pth) to transformers' LlamaForCausalLM.

The architecture is the same: RoPE (rotate-half, same frequencies), RMSNorm, SwiGLU, no biases
in q/k/v or the MLP, tied embeddings as in config.py. One difference: model.py's attention output
projection has a bias. Llama's attention_bias switches on all four q/k/v/o biases together, so
q/k/v get zero biases and the forward pass is unchanged.

The export is saved with the SmolLM2 tokenizer, then reloaded from disk and checked: the same
input must give logits within 1e-3 (fp32) of the original model, or the script fails.

    uv run python export_hf.py --checkpoint model.pth --model 124m --out exports/fw60_py20_fm13_gb07_s1234_124m
    uv run python export_hf.py --random --out exports/random_init_124m    # untrained, for sanity checks

--model picks the config.py MODEL_CONFIGS entry the checkpoint was trained with (a wrong one fails
on the shape check when loading); use train.py's printed run name as the --out directory.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from config import MODEL_CONFIGS
from model import GPTModel
from tokenize_data import TOKENIZER

TOLERANCE = 1e-3


def llama_config(cfg, ffn_hidden_dim):
    if cfg.ffn != "swiglu" or cfg.norm != "rmsnorm" or cfg.qkv_bias:
        raise SystemExit("export needs ffn=swiglu, norm=rmsnorm, qkv_bias=False")
    return LlamaConfig(
        vocab_size=cfg.padded_vocab_size,
        hidden_size=cfg.emb_dim,
        intermediate_size=ffn_hidden_dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.context_length,
        rms_norm_eps=1e-5,  # nn.RMSNorm(eps=1e-5) in model.make_norm
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_base},
        attention_bias=True,  # only o_proj's is non-zero, see the docstring
        mlp_bias=False,
        tie_word_embeddings=cfg.tie_embeddings,
        bos_token_id=0,
        eos_token_id=0,
    )


def convert(gpt, cfg):
    """LlamaForCausalLM with gpt's weights."""
    llama = LlamaForCausalLM(llama_config(cfg, gpt.transformer_blocks[0].ff.hidden_dim))
    sd = {"model.embed_tokens.weight": gpt.token_emb.weight,
          "model.norm.weight": gpt.final_norm.weight,
          "lm_head.weight": gpt.out_head.weight}
    for i, block in enumerate(gpt.transformer_blocks):
        p = f"model.layers.{i}."
        att = block.att
        sd |= {
            p + "input_layernorm.weight": block.norm1.weight,
            p + "post_attention_layernorm.weight": block.norm2.weight,
            p + "self_attn.q_proj.weight": att.W_query.weight,
            p + "self_attn.k_proj.weight": att.W_key.weight,
            p + "self_attn.v_proj.weight": att.W_value.weight,
            p + "self_attn.o_proj.weight": att.out_proj.weight,
            p + "self_attn.o_proj.bias": att.out_proj.bias,
            p + "self_attn.q_proj.bias": torch.zeros(cfg.emb_dim),
            p + "self_attn.k_proj.bias": torch.zeros(cfg.emb_dim),
            p + "self_attn.v_proj.bias": torch.zeros(cfg.emb_dim),
            p + "mlp.gate_proj.weight": block.ff.w_gate.weight,
            p + "mlp.up_proj.weight": block.ff.w_up.weight,
            p + "mlp.down_proj.weight": block.ff.w_down.weight,
        }
    missing, unexpected = llama.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    # With tied embeddings lm_head.weight is the embedding, so it may be reported as missing.
    missing = [k for k in missing if not (cfg.tie_embeddings and k == "lm_head.weight")]
    if missing or unexpected:
        raise SystemExit(f"weight mapping incomplete: missing {missing}, unexpected {unexpected}")
    return llama


@torch.no_grad()
def check_logits(gpt, llama, cfg):
    """Max |logit difference| on random tokens, both models in fp32 on CPU."""
    gpt.eval()
    llama.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), generator=torch.Generator().manual_seed(0))
    diff = (gpt(ids) - llama(ids).logits).abs().max().item()
    return diff


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="GPTModel state_dict, e.g. model.pth")
    source.add_argument("--random", action="store_true", help="randomly initialized model (seed 123)")
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="124m", help="config.py MODEL_CONFIGS name")
    parser.add_argument("--out", required=True, help="output directory, e.g. exports/<run_name>")
    args = parser.parse_args()

    torch.manual_seed(123)
    cfg = MODEL_CONFIGS[args.model]
    gpt = GPTModel(cfg)
    if args.checkpoint:
        gpt.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location="cpu"))
    gpt.float()

    llama = convert(gpt, cfg)
    out = Path(args.out)
    llama.save_pretrained(out)
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(out)

    # Check the saved files, not the in-memory copy.
    reloaded = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32)
    diff = check_logits(gpt, reloaded, cfg)
    print(f"max |logit diff| GPTModel vs exported Llama: {diff:.2e} (tolerance {TOLERANCE:.0e})")
    if not diff < TOLERANCE:
        raise SystemExit(f"FAILED: exported model does not match ({diff:.2e} >= {TOLERANCE:.0e})")
    print(f"Saved {out}/ ({sum(p.numel() for p in reloaded.parameters()) / 1e6:.1f}M parameters)")


if __name__ == "__main__":
    main()
