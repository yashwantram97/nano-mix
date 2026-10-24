import math

import numpy as np
import pytest
import torch

from nanomix.eval.bpb import BPB_WINDOW, evaluate_bpb, token_byte_lengths
from nanomix.model.gpt import GPTModel


def test_random_init_bpb_is_near_uniform(tiny_config):
    """An untrained model guesses ~uniformly: log2(vocab) bits per scored token, over their bytes."""
    rng = np.random.default_rng(0)
    byte_lengths = token_byte_lengths()
    ordinary = np.flatnonzero(byte_lengths > 0)  # no special tokens, as in real text
    tokens = rng.choice(ordinary, 2 * BPB_WINDOW + 1).astype(np.uint16)

    torch.manual_seed(0)
    bpb = evaluate_bpb(GPTModel(tiny_config), {"random": tokens}, batch_size=2)["random"]
    scored = tokens[1:]
    uniform = math.log2(tiny_config.vocab_size) * len(scored) / byte_lengths[scored].sum()
    assert bpb == pytest.approx(uniform, rel=0.02)


def test_bpb_refuses_a_short_context(tiny_config):
    with pytest.raises(SystemExit, match="BPB_WINDOW"):
        evaluate_bpb(GPTModel(tiny_config.replace(context_length=256)), {}, batch_size=1)
