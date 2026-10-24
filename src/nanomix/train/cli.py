# Copyright (c) Sebastian Raschka under Apache License 2.0.
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Train the GPT on the mix: set up the model, data, logging and checkpoints, then run the loop.

    uv run nanomix-train --model 124m --preset h100 2>&1 | tee -ia logs/train_124m.log
    uv run nanomix-train --model 124m --preset h100 --resume checkpoints/<run_id>/ckpt.pt ...
    uv run nanomix-train --preset mac --max-steps 20 --set warmup_steps=2 --tag smoke

--model picks a MODEL_CONFIGS entry (model/config.py), --preset a TRAIN_PRESETS entry
(train/settings.py); --max-steps and --set override single settings.
"""

import argparse
import ast
import dataclasses
import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from nanomix.data.loader import create_mix_dataloaders
from nanomix.eval.bpb import BPB_WINDOW, load_eval_sets
from nanomix.model.config import MODEL_CONFIGS, model_label
from nanomix.model.gpt import GPTModel
from nanomix.sources import SOURCES
from nanomix.train.checkpoint import Checkpointer, StopRequest, check_resume, load_checkpoint, to_cpu
from nanomix.train.loop import load_val_batches, train
from nanomix.train.metrics import (
    MetricsLog,
    init_wandb,
    resolve_peak_flops,
    save_loss_plot,
    training_flops_per_token,
)
from nanomix.train.optim import param_groups
from nanomix.train.precision import autocast_ctx, resolve_device, resolve_precision
from nanomix.train.settings import TRAIN_PRESETS


def run_name(mix_cfg, gpt_config):
    """e.g. fw60_py20_fm13_gb07_s1234_124m: mix ratios in percent, the mix seed, the model size."""
    parts = [f"{SOURCES[n].abbrev if n in SOURCES else n}{round(r * 100):02d}"
             for n, r in mix_cfg["sources"].items()]
    return "_".join(parts) + f"_s{mix_cfg['seed']}_{model_label(gpt_config)}"


def run(gpt_config, settings, resume=None, tag=None):
    """Train a new run, or continue the one in the checkpoint at ``resume``.

    The run id is the run name (+ "_<tag>"); checkpoints go to <checkpoint_dir>/<run_id>/ and
    metrics to <metrics_dir>/<run_id>.jsonl. A new run refuses to start over an existing one.
    """
    load_dotenv()  # WANDB_API_KEY etc. from .env (gitignored); doesn't override the environment
    torch.manual_seed(123)
    device = resolve_device()

    precision = resolve_precision(settings.precision, device)
    amp = autocast_ctx(device, precision)
    print(f"Using device: {device} ({precision})")

    # Model and optimizer, restored from the checkpoint on a resume.
    ckpt = load_checkpoint(resume) if resume else None
    if ckpt and ckpt["model_config"] != dataclasses.asdict(gpt_config):
        raise SystemExit(f"Can't resume: {resume} was trained with a different model config (--model)")
    model = GPTModel(gpt_config)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    model.to(device)
    train_model = model
    if settings.compile and device.type == "cuda":
        # A compiled wrapper sharing model's weights. Training steps go through it; sampling and
        # holdout BPB use model itself, so their changing shapes never recompile the training
        # graph. state_dict keys in checkpoints stay those of model (no _orig_mod. prefix).
        train_model = torch.compile(model)
        # Print every recompile. Expected: one or two at the first val loss (eval mode, no_grad).
        torch._logging.set_logs(recompiles=True)
        print("Compiling model (first step is slow)")
    groups = param_groups(model, settings.weight_decay)
    # fused: one kernel for the whole update instead of several per parameter (CUDA only).
    optimizer = torch.optim.AdamW(groups, lr=settings.learning_rate, betas=settings.adam_betas,
                                  fused=device.type == "cuda")
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])  # moves the state to the params' device
        torch.set_rng_state(ckpt["rng"]["torch"])
        if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    start_step = ckpt["step"] if ckpt else 0
    print(f"AdamW: decay {settings.weight_decay} on {sum(p.numel() for p in groups[0]['params']) / 1e6:.1f}M "
          f"params, none on {sum(p.numel() for p in groups[1]['params']) / 1e6:.1f}M (norms, biases, embedding)")

    # Data: the OLMo-core mix, built the same way as mix.py / verify_mix.py. A resumed run skips
    # the instances it already trained on (the order is a fixed seeded permutation).
    train_loader, val_loader, mix_cfg = create_mix_dataloaders(
        settings.mix_config,
        batch_size=settings.batch_size,
        val_instances=settings.val_instances,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        skip_instances=start_step * settings.grad_accum_steps * settings.batch_size,
    )
    if ckpt:
        check_resume(ckpt, gpt_config, settings, mix_cfg)
    if mix_cfg["sequence_length"] != gpt_config.context_length:
        raise SystemExit(f"mix sequence_length {mix_cfg['sequence_length']} != "
                         f"model context_length {gpt_config.context_length}")
    if gpt_config.context_length < BPB_WINDOW:
        raise SystemExit(f"model context_length {gpt_config.context_length} < BPB_WINDOW {BPB_WINDOW}: "
                         "holdout BPB is scored on fixed 1024-token windows")
    remaining = settings.max_steps - start_step
    if remaining * settings.grad_accum_steps > len(train_loader):
        raise SystemExit(f"{remaining} steps x {settings.grad_accum_steps} accum needs "
                         f"{remaining * settings.grad_accum_steps:,} micro-batches; the mix "
                         f"has {len(train_loader):,} left. Lower max_steps (no repetition).")

    tokenizer = AutoTokenizer.from_pretrained(mix_cfg["tokenizer"])
    if len(tokenizer) != gpt_config.vocab_size:
        raise SystemExit(f"tokenizer has {len(tokenizer)} tokens, model vocab_size is {gpt_config.vocab_size}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters")

    tokens_per_step = settings.tokens_per_optimizer_step(gpt_config.context_length)
    print(f"{len(train_loader):,} micro-batches left in the mix; "
          f"running steps {start_step + 1:,}-{settings.max_steps:,}")
    print(f"Effective batch: {settings.batch_size} x {settings.grad_accum_steps} accum "
          f"x {gpt_config.context_length} ctx = {tokens_per_step:,} tokens/step "
          f"({tokens_per_step * settings.max_steps:,} tokens total)")
    if tokens_per_step < 50_000:
        print(f"  WARNING: {tokens_per_step:,} tokens/step is small. GPT-2 small was "
              f"pretrained near 500k; gradients this noisy make the loss curve and "
              f"LR schedule hard to reason about. Raise batch_size or "
              f"grad_accum_steps.")

    # The run config, printed once (mix ratios, seed, model size, LR, ...).
    name = ckpt["run_id"] if ckpt else run_name(mix_cfg, gpt_config) + (f"_{tag}" if tag else "")
    print(f"Run: {name}" + (f" (resuming after step {start_step:,})" if ckpt else ""))
    run_dir = Path(settings.checkpoint_dir) / name
    metrics_path = Path(settings.metrics_dir) / f"{name}.jsonl"
    if not ckpt and ((run_dir / "ckpt.pt").exists() or metrics_path.exists()):
        raise SystemExit(f"{run_dir}/ckpt.pt or {metrics_path} already exists. Resume it with "
                         f"--resume {run_dir / 'ckpt.pt'}, pass --tag for a separate run, or delete them.")
    print("  " + json.dumps({
        "mix": mix_cfg["sources"], "mix_seed": mix_cfg["seed"],
        "mix_tokens": mix_cfg["requested_tokens"], "params_M": round(n_params / 1e6, 1),
        "model": dataclasses.asdict(gpt_config), "train": dataclasses.asdict(settings),
    }))

    eval_sets = load_eval_sets()
    print(f"Holdout eval sets: {', '.join(f'{n} ({len(t):,} tokens)' for n, t in eval_sets.items())}; "
          f"every {settings.eval_every_tokens:,} tokens and at the end")
    val_batches = load_val_batches(val_loader, settings.eval_iter, device)
    print(f"Val loss: {len(val_batches)} fixed batches ({sum(x.numel() for x, _ in val_batches):,} tokens) "
          f"every {settings.eval_freq} steps")

    flops_per_token = training_flops_per_token(gpt_config, n_params)
    peak_flops = resolve_peak_flops(device, settings)
    if peak_flops:
        print(f"MFU: {flops_per_token / 1e9:.2f} GFLOP/token, peak {peak_flops / 1e12:.0f} TFLOP/s "
              f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'set in settings'})")
    else:
        print("MFU: GPU peak unknown (set peak_tflops), not printed")

    wandb_run = init_wandb(settings, name, ckpt.get("wandb_id") if ckpt else None, {
        "run_id": name, "n_params": n_params, "flops_per_token": flops_per_token,
        "peak_flops": peak_flops, "model": dataclasses.asdict(gpt_config),
        "train": dataclasses.asdict(settings), "mix": mix_cfg})
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}")
    metrics = MetricsLog(metrics_path, wandb_run, run_id=name)
    metrics("resume" if ckpt else "run", run_name=name, step=start_step, n_params=n_params,
            flops_per_token=flops_per_token, peak_flops=peak_flops,
            model=dataclasses.asdict(gpt_config), train=dataclasses.asdict(settings), mix=mix_cfg)
    print(f"Metrics: appended to {metrics_path}")

    checkpointer = Checkpointer(run_dir, settings.checkpoint_minutes, settings.checkpoint_s3,
                                extra_uploads=[metrics_path])
    print(f"Checkpoints: {run_dir}/ every {settings.checkpoint_minutes} min, on Ctrl-C / SIGTERM"
          f"{' / spot notice' if settings.watch_spot else ''} and at the end"
          + (f"; copied to {checkpointer.s3}/" if checkpointer.s3 else ""))

    def make_state(step, tokens_seen, history):
        return {
            "run_id": name, "step": step, "tokens_seen": tokens_seen,
            "model": to_cpu(model.state_dict()), "optimizer": to_cpu(optimizer.state_dict()),
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
            "history": history, "model_config": dataclasses.asdict(gpt_config),
            "train_settings": dataclasses.asdict(settings), "mix": mix_cfg,
            "wandb_id": wandb_run.id if wandb_run is not None else None,
        }

    history, finished = train(
        model, train_model, train_loader, val_batches, optimizer, device, settings, tokenizer,
        amp=amp, eval_sets=eval_sets, flops_per_token=flops_per_token, peak_flops=peak_flops,
        metrics=metrics, start_step=start_step,
        tokens_seen=ckpt["tokens_seen"] if ckpt else 0, history=ckpt["history"] if ckpt else None,
        checkpointer=checkpointer, make_state=make_state, stop=StopRequest(settings.watch_spot),
    )
    metrics.finish()
    # One plot per run, next to its metrics file (a resume redraws it from the full history).
    plot_path = Path(settings.metrics_dir) / f"{name}_loss.pdf"
    save_loss_plot(history, plot_path)
    print(f"Loss plot: {plot_path}")
    if finished:
        print(f"Done. Final checkpoint: {checkpointer.path} "
              f"(weights: torch.load(path, weights_only=True)['model'])")


def parse_settings(preset, max_steps, overrides):
    """The preset with --max-steps and --set KEY=VALUE applied together, so e.g. a shorter run
    and a shorter warmup are validated as one change."""
    settings = TRAIN_PRESETS[preset]
    changes = {"max_steps": max_steps} if max_steps else {}
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in {f.name for f in dataclasses.fields(settings)}:
            raise SystemExit(f"--set {item}: TrainSettings has no field {key!r}")
        try:
            changes[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            changes[key] = value  # a bare string, e.g. checkpoint_dir=foo
    try:
        return settings.replace(**changes)
    except ValueError as e:
        raise SystemExit(f"Invalid settings: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="124m", help="model/config.py MODEL_CONFIGS name")
    parser.add_argument("--preset", choices=TRAIN_PRESETS, default="mac",
                        help="train/settings.py TRAIN_PRESETS name: mac (quick test) or h100 (the 5B run)")
    parser.add_argument("--max-steps", type=int, help="override the preset's max_steps (smoke runs)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override TrainSettings fields, e.g. --set warmup_steps=20 batch_size=64 "
                             "grad_accum_steps=8 (values are Python literals)")
    parser.add_argument("--tag", help="suffix for the run id, e.g. smoke: a separate run with its own "
                                      "checkpoints and metrics")
    parser.add_argument("--resume", help="continue from this checkpoint (checkpoints/<run_id>/ckpt.pt); "
                                         "--model, --preset, --max-steps and --set must match it")
    args = parser.parse_args()
    # Print each line as it is written, also into a pipe (| tee), as python -u did.
    sys.stdout.reconfigure(line_buffering=True)
    run(MODEL_CONFIGS[args.model], parse_settings(args.preset, args.max_steps, args.set),
        resume=args.resume, tag=args.tag)


if __name__ == "__main__":
    main()
