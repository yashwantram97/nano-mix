"""Shared fixtures. Tests are small and CPU only; the SmolLM2 tokenizer comes from the local HF cache."""

import os

# Before transformers / huggingface_hub are imported: never reach the network (CLAUDE.md).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

from nanomix.model.config import ModelConfig  # noqa: E402


@pytest.fixture
def tiny_config():
    """A 2-layer model with the real vocab and the 1024 context that holdout BPB needs."""
    return ModelConfig(context_length=1024, emb_dim=64, n_heads=2, n_layers=2)
