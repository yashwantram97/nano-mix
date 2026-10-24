import pytest

from nanomix.model.gpt import GPTModel
from nanomix.train.optim import lr_at_step, param_groups
from nanomix.train.settings import H100_SETTINGS


def test_lr_warmup_peak_and_cosine_end():
    s = H100_SETTINGS  # warmup 300 to 6e-4, cosine to 6e-5 at step 9,617
    assert lr_at_step(1, s) == pytest.approx(6e-4 / 300)
    assert lr_at_step(150, s) == pytest.approx(3e-4)
    assert lr_at_step(300, s) == pytest.approx(6e-4)
    assert lr_at_step(s.max_steps, s) == pytest.approx(6e-5)
    middle = 300 + (s.max_steps - 300) / 2
    assert lr_at_step(middle, s) == pytest.approx((6e-4 + 6e-5) / 2)
    lrs = [lr_at_step(step, s) for step in range(300, s.max_steps + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # never rises after warmup


def test_lr_constant_schedule():
    s = H100_SETTINGS.replace(lr_schedule="constant")
    assert lr_at_step(s.max_steps, s) == pytest.approx(6e-4)


def test_no_weight_decay_on_norms_biases_and_embedding(tiny_config):
    model = GPTModel(tiny_config)
    decay, no_decay = param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0

    decayed = {id(p) for p in decay["params"]}
    assert model.out_head.weight is model.token_emb.weight  # tied: the head gets no decay either
    for name, p in model.named_parameters():  # a tied matrix appears once
        no_decay_expected = "norm" in name or name.endswith("bias") or name == "token_emb.weight"
        assert (id(p) in decayed) != no_decay_expected, name
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
