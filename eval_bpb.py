"""Bits per byte (BPB) on the frozen holdout sets in data/eval/ (made by make_eval_sets.py).

Each eval stream is cut into non-overlapping seq_len windows; window i reads tokens
[i*L, (i+1)*L) and is scored on the next tokens [i*L+1, (i+1)*L+1), so every token after the
first is scored exactly once. Over the whole set:

    BPB = sum of loss in nats / ln(2) / sum of UTF-8 bytes of the scored tokens

summed first and divided once (no per-window averaging). A token's byte count comes from the
tokenizer's byte-level mapping, so a token holding part of a multi-byte character counts only
its own bytes; EOS and other special tokens count 0. The SmolLM2 vocab has no token for 21 byte
values (mostly ASCII control characters such as 0x13), so tokenizing drops them -- in the training
data too. They are never predicted and not counted (9 of finemath's 6.3M bytes). Loss is computed in fp32 (no autocast), so
BPB is comparable across runs.

    uv run python eval_bpb.py --check-bytes          # byte table == len(text.encode("utf-8"))
    uv run python eval_bpb.py --random               # randomly initialized model (sanity check)
    uv run python eval_bpb.py --checkpoint model.pth
"""

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from tokenize_data import TOKENIZER, get_tokenizer

EVAL = Path("data/eval")
RAW_EVAL = Path("data/raw_eval")


@functools.cache
def token_byte_lengths():
    """UTF-8 byte length of every token id, from the byte-level BPE alphabet. Special tokens are 0."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    tok = get_tokenizer()
    char_to_byte = {c: b for b, c in bytes_to_unicode().items()}
    lengths = np.zeros(len(tok), dtype=np.int64)
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    for token, i in tok.get_vocab().items():
        if i not in special:
            assert all(c in char_to_byte for c in token), f"token {i} {token!r} is not byte-level"
            lengths[i] = len(token)  # one alphabet character per byte
    return lengths


def load_eval_sets(names=None):
    """{name: uint16 token array} for every set in data/eval/ (or just `names`)."""
    names = names or sorted(p.stem for p in EVAL.glob("*.json"))
    sets = {}
    for name in names:
        meta = json.loads((EVAL / f"{name}.json").read_text())
        assert meta["tokenizer"] == TOKENIZER, f"{name} was tokenized with {meta['tokenizer']}"
        sets[name] = np.fromfile(EVAL / f"{name}.npy", dtype=np.uint16)
    return sets


@torch.no_grad()
def evaluate_bpb(model, eval_sets, seq_len, batch_size):
    """{name: BPB} for a GPTModel. Restores the model's train/eval mode afterwards."""
    device = next(model.parameters()).device
    byte_lengths = token_byte_lengths()
    was_training = model.training
    model.eval()
    results = {}
    for name, tokens in eval_sets.items():
        num_windows = (len(tokens) - 1) // seq_len
        nats = 0.0
        num_bytes = 0
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            chunk = tokens[start * seq_len : end * seq_len + 1].astype(np.int64)
            targets = chunk[1:]
            x = torch.from_numpy(chunk[:-1]).view(-1, seq_len).to(device)
            y = torch.from_numpy(targets).to(device)
            # Drop padded vocab rows (never valid targets) so the softmax is over the real vocab.
            logits = model(x)[..., : model.vocab_size].float()
            nats += F.cross_entropy(logits.flatten(0, 1), y, reduction="sum").item()
            num_bytes += int(byte_lengths[targets].sum())
        results[name] = nats / math.log(2) / num_bytes
    model.train(was_training)
    return results


def unencodable_bytes():
    """Byte values the vocab has no token for; the tokenizer drops them."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    vocab = get_tokenizer().get_vocab()
    return [b for b, c in bytes_to_unicode().items() if c not in vocab]


def check_bytes(num_docs=100):
    """Summed byte table == len(text.encode("utf-8")) on the first num_docs docs of each set, and on
    each whole set against its .json num_bytes, less the bytes the vocab cannot encode."""
    tok = get_tokenizer()
    byte_lengths = token_byte_lengths()
    dropped_values = unencodable_bytes()
    print(f"  bytes with no token in the vocab: {' '.join(f'{b:#04x}' for b in sorted(dropped_values))}")
    ok = True
    for name, tokens in load_eval_sets().items():
        texts = pq.read_table(RAW_EVAL / f"{name}.parquet", columns=["text"]).column("text").to_pylist()
        bad = 0
        for text in texts[:num_docs]:
            ids = tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            raw = text.encode("utf-8")
            bad += int(byte_lengths[ids].sum()) != len(raw) - sum(raw.count(b) for b in dropped_values)
        raw = b"".join(t.encode("utf-8") for t in texts)
        dropped = sum(raw.count(b) for b in dropped_values)
        whole = int(byte_lengths[tokens].sum())
        expected = json.loads((EVAL / f"{name}.json").read_text())["num_bytes"]
        passed = bad == 0 and whole == expected - dropped
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:13s} {min(num_docs, len(texts))} docs, {bad} mismatched; "
              f"whole set {whole:,} bytes = {expected:,} in .json - {dropped} unencodable")
    if not ok:
        raise SystemExit("byte table check FAILED")


def main():
    from config import MODEL_CONFIG
    from model import GPTModel
    from train import resolve_device

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-bytes", action="store_true")
    group.add_argument("--random", action="store_true", help="randomly initialized model")
    group.add_argument("--checkpoint", help="GPTModel state_dict, e.g. model.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.check_bytes:
        check_bytes()
        return

    torch.manual_seed(123)
    model = GPTModel(MODEL_CONFIG)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(resolve_device())
    seq_len = MODEL_CONFIG.context_length

    sets = load_eval_sets()
    bpb = evaluate_bpb(model, sets, seq_len, args.batch_size)
    byte_lengths = token_byte_lengths()
    print(f"{'set':13s} {'BPB':>7s} {'uniform':>8s}")
    for name, tokens in sets.items():
        # A uniform guess costs log2(vocab) bits per scored token.
        scored = tokens[1 : (len(tokens) - 1) // seq_len * seq_len + 1]
        uniform = math.log2(MODEL_CONFIG.vocab_size) * len(scored) / byte_lengths[scored].sum()
        print(f"{name:13s} {bpb[name]:7.3f} {uniform:8.3f}")


if __name__ == "__main__":
    main()
