"""AdamW parameter groups and the learning-rate schedule."""

import math


def param_groups(model, weight_decay):
    """AdamW groups: decay on weight matrices; none on norms, biases or the embedding.

    With tied embeddings the embedding is also the output head; it gets no decay either way.
    """
    embedding = model.token_emb.weight
    decay, no_decay = [], []
    for p in model.parameters():  # yields a tied matrix once
        (decay if p.ndim >= 2 and p is not embedding else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at_step(step, settings):
    """Learning rate for a 1-based optimizer step.

    Linear warmup, then cosine decay to ``min_lr_ratio * learning_rate``.
    A pure function of the step so it can be tested without a training run.
    """
    peak = settings.learning_rate
    warmup = settings.warmup_steps

    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    if settings.lr_schedule == "constant":
        return peak

    progress = (step - warmup) / max(1, settings.max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = peak * settings.min_lr_ratio
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
