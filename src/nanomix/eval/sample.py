# Copyright (c) Sebastian Raschka under Apache License 2.0.
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Greedy text samples from fixed prompts: a quick human check during training."""

import contextlib

import torch

from nanomix.model.gpt import mask_padded_logits

# One prompt per kind of data in the mix. Greedy, so repetition early in a run is normal.
SAMPLE_PROMPTS = {
    "prose": "The history of the printing press",
    "code": "def fibonacci(n):\n",
    "math": "Solve for x: 3x + 7 = 22.\nSolution:",
}


@torch.no_grad()
def greedy_generate(model, idx, max_new_tokens):
    """Append max_new_tokens argmax tokens to idx (B, T), cropping the context to the model's."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.context_length:])
        # Padded vocab rows decode to nothing, so they must never be selected.
        logits = mask_padded_logits(logits[:, -1, :], model.vocab_size)
        idx = torch.cat((idx, torch.argmax(logits, dim=-1, keepdim=True)), dim=1)
    return idx


def print_samples(model, tokenizer, device, amp=None, max_new_tokens=100):
    """Print a greedy continuation of every SAMPLE_PROMPTS prompt."""
    was_training = model.training
    model.eval()
    for name, prompt in SAMPLE_PROMPTS.items():
        ids = tokenizer.encode(prompt, add_special_tokens=False)  # as in tokenize_data.py
        with amp if amp is not None else contextlib.nullcontext():
            out = greedy_generate(model, torch.tensor([ids], device=device), max_new_tokens)
        print(f"[sample/{name}] {tokenizer.decode(out[0].tolist())!r}")
    model.train(was_training)
