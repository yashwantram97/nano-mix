import pytest
import torch

from nanomix.model.config import MODEL_CONFIGS
from nanomix.model.gpt import GPTModel


def test_output_shape(tiny_config):
    model = GPTModel(tiny_config)
    out = model(torch.randint(0, tiny_config.vocab_size, (2, 16)))
    assert out.shape == (2, 16, tiny_config.padded_vocab_size)


@pytest.mark.parametrize("name, millions", [("40m", 40.1), ("124m", 122.7)])
def test_parameter_count(name, millions):
    with torch.device("meta"):  # shapes only, no memory
        model = GPTModel(MODEL_CONFIGS[name])
    n_params = sum(p.numel() for p in model.parameters())  # the tied embedding counts once
    assert round(n_params / 1e6, 1) == millions
