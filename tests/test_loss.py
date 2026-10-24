import numpy as np
import torch
import torch.nn.functional as F

from nanomix.data.loader import IGNORE_INDEX, MixInstances
from nanomix.model.gpt import GPTModel
from nanomix.train.loop import batch_loss


def test_last_position_is_not_scored(tiny_config):
    seq_len = 32
    tokens = np.arange(3 * seq_len) % 1000
    mix = [{"input_ids": tokens[i * seq_len:(i + 1) * seq_len]} for i in range(3)]
    x, y = MixInstances(mix, [0, 2])[1]  # instance 2 of the mix

    assert torch.equal(x, torch.as_tensor(tokens[2 * seq_len:], dtype=torch.long))
    assert torch.equal(y[:-1], x[1:])  # the next token
    assert y[-1] == IGNORE_INDEX
    assert (y != IGNORE_INDEX).sum() == seq_len - 1

    torch.manual_seed(0)
    model = GPTModel(tiny_config)
    loss = batch_loss(x[None], y[None], model, torch.device("cpu"))
    with torch.no_grad():
        logits = model(x[None])[0]
    expected = F.cross_entropy(logits[:-1], x[1:])  # mean over the first seq_len - 1 positions only
    assert torch.allclose(loss, expected, atol=1e-5)
