# Copyright (c) Sebastian Raschka under Apache License 2.0.
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""The training loop: optimizer steps, val loss, holdout BPB, samples and checkpoints."""

import contextlib
import math
import time

import torch

from nanomix.eval.bpb import evaluate_bpb
from nanomix.eval.sample import print_samples
from nanomix.train.optim import lr_at_step


def batch_loss(input_batch, target_batch, model, device, amp=None):
    """Mean next-token cross-entropy of one batch."""
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    with amp if amp is not None else contextlib.nullcontext():
        logits = model(input_batch)
        # cross_entropy upcasts to fp32 internally under autocast, so the
        # softmax and its gradient are not computed in bf16. Targets equal to
        # -100 (the last position of each instance, loader.IGNORE_INDEX) are
        # skipped, and the mean is over the scored targets only.
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten())
    return loss


def load_val_batches(val_loader, num_batches, device):
    """The first num_batches val batches, on the device. Read once; every val loss uses them."""
    batches = []
    for input_batch, target_batch in val_loader:
        if len(batches) == num_batches:
            break
        batches.append((input_batch.to(device), target_batch.to(device)))
    return batches


def evaluate_val_loss(model, val_batches, device, amp=None):
    """Mean loss over the fixed val batches. One GPU sync, at the end."""
    model.eval()
    with torch.no_grad():
        total = sum(batch_loss(x, y, model, device, amp) for x, y in val_batches)
    model.train()
    return (total / len(val_batches)).item()


def holdout_eval(model, eval_sets, settings, step, tokens_seen, metrics=None):
    """Holdout BPB on every frozen eval set, printed and logged."""
    t0 = time.time()
    bpb = evaluate_bpb(model, eval_sets, settings.batch_size)
    print(f"[eval {tokens_seen:,} tokens] " + "  ".join(f"bpb/{n} {v:.4f}" for n, v in bpb.items())
          + f"  ({time.time() - t0:.0f}s)")
    if metrics:
        metrics("bpb", step=step, tokens=tokens_seen, **{f"bpb/{n}": v for n, v in bpb.items()})


def train(model, train_model, train_loader, val_batches, optimizer, device, settings, tokenizer,
          amp=None, eval_sets=None, flops_per_token=None, peak_flops=None, metrics=None, start_step=0,
          tokens_seen=0, history=None, checkpointer=None, make_state=None, stop=None):
    """Train for ``settings.max_steps`` optimizer steps.

    One optimizer step consumes ``grad_accum_steps`` micro-batches. Their
    gradients sum into ``.grad``, so the update is the same one a single batch
    of ``batch_size * grad_accum_steps`` would have produced -- at the memory
    cost of one micro-batch, and for the same total FLOPs.

    ``train_model`` is the (possibly compiled) model the steps run through; ``model`` is the
    same weights uncompiled, used for sampling and holdout BPB so their varying shapes don't
    recompile the training graph. Losses and grad norms stay on the GPU between log lines:
    calling ``.item()`` per micro-batch would make the CPU wait for the GPU every time, so it
    could never queue the next micro-batch ahead.

    A resumed run passes ``start_step``, ``tokens_seen`` and the loss ``history`` from its
    checkpoint (and a train_loader that starts at the next unseen instance). After each step,
    a checkpoint is saved when ``checkpointer`` says one is due, or when ``stop`` is set (Ctrl-C,
    SIGTERM, spot notice), in which case training then returns early. Returns
    (history, finished): history holds the val-loss points (train_losses, val_losses, steps, tokens).
    """
    history = history or {"train_losses": [], "val_losses": [], "steps": [], "tokens": []}

    max_steps = settings.max_steps
    accum_steps = settings.grad_accum_steps
    clip = settings.grad_clip
    sample_freq = settings.sample_freq or settings.eval_freq * 10
    cuda = device.type == "cuda"

    # The first holdout eval threshold above tokens_seen (0 for a new run).
    next_eval = (tokens_seen // settings.eval_every_tokens + 1) * settings.eval_every_tokens
    zero = torch.zeros((), device=device)
    # Sums over the steps since the last log line (GPU tensors) / since the last val loss.
    log_loss, log_grad_norm, log_clipped = zero.clone(), zero.clone(), zero.clone()
    log_tokens, log_steps = 0, 0
    train_loss_sum, train_loss_steps = 0.0, 0
    log_start = time.time()
    # Slowdown alert: tok/s under half the best line so far, ignoring the first lines (compiling).
    best_tokens_per_sec, log_lines = 0.0, 0

    train_model.train()
    # One pass: the caller checks the mix holds max_steps of batches, so this never
    # runs out and no instance is seen twice.
    batches = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, max_steps + 1):
        # The schedule is indexed by optimizer step, not micro-batch.
        lr = lr_at_step(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(accum_steps):
            input_batch, target_batch = next(batches)
            loss = batch_loss(input_batch, target_batch, train_model, device, amp)
            # Divide so the accumulated gradient is the *mean* over the full
            # effective batch. Omitting this scales gradients by accum_steps,
            # which silently acts like an accum_steps-times larger learning
            # rate -- it does not error, it just diverges.
            (loss / accum_steps).backward()
            log_loss += loss.detach() / accum_steps
            log_tokens += input_batch.numel()
        tokens_seen += accum_steps * input_batch.numel()

        # Clip once, on the accumulated (true full-batch) gradient. clip_grad_norm_ returns
        # the norm before clipping; with clipping off, an infinite max_norm only measures it.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf"))
        log_grad_norm += grad_norm
        if clip:
            log_clipped += grad_norm > clip
        log_steps += 1

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % settings.log_freq == 0 or step == max_steps:
            # .item() waits for the GPU, so the time is taken after it.
            loss_mean = log_loss.item() / log_steps
            grad_norm_mean = log_grad_norm.item() / log_steps
            clip_frac = log_clipped.item() / log_steps
            elapsed = time.time() - log_start
            tokens_per_sec = log_tokens / elapsed
            record = {"step": step, "tokens": tokens_seen, "loss": loss_mean, "lr": lr,
                      "grad_norm": grad_norm_mean, "clip_frac": clip_frac,
                      "tok_per_s": tokens_per_sec, "step_s": elapsed / log_steps}
            line = (f"Step {step:06d}/{max_steps} ({tokens_seen:,} tokens): loss {loss_mean:.3f}, "
                    f"lr {lr:.2e}, grad norm {grad_norm_mean:.2f}, clip frac {clip_frac:.2f}, "
                    f"{tokens_per_sec:,.0f} tok/s, {elapsed / log_steps:.2f} s/step")
            if peak_flops and flops_per_token:
                record["mfu"] = tokens_per_sec * flops_per_token / peak_flops
                line += f", MFU {record['mfu']:.1%}"
            if cuda:
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                line += f", peak mem {record['peak_mem_gb']:.1f} GB"
            print(line)
            if metrics:
                metrics("train", **record)
            if not math.isfinite(loss_mean):
                # Stop without saving: the newest weights are likely broken, and a save would
                # push the last good checkpoint out to ckpt_prev.pt.
                if metrics:
                    metrics.alert("Loss is not finite", f"{metrics.run_id}: loss {loss_mean} at step {step}; "
                                  "stopped without saving a checkpoint")
                raise SystemExit(f"Loss is {loss_mean} at step {step}; stopped. Resume from the last checkpoint "
                                 "(lower the learning rate or check the data).")
            log_lines += 1
            if log_lines > 3 and tokens_per_sec < 0.5 * best_tokens_per_sec and metrics:
                metrics.alert("Training slowed down", f"{metrics.run_id}: {tokens_per_sec:,.0f} tok/s at step "
                              f"{step}, best so far {best_tokens_per_sec:,.0f}")
            best_tokens_per_sec = max(best_tokens_per_sec, tokens_per_sec if log_lines > 1 else 0.0)
            train_loss_sum += loss_mean * log_steps
            train_loss_steps += log_steps
            for total in (log_loss, log_grad_norm, log_clipped):
                total.zero_()
            log_tokens, log_steps = 0, 0
            log_start = time.time()

        paused = time.time()
        if step % settings.eval_freq == 0 or step == max_steps:
            # Train loss: the mean over the steps since the last val loss (no extra forward passes).
            train_loss = train_loss_sum / max(train_loss_steps, 1)
            val_loss = evaluate_val_loss(train_model, val_batches, device, amp)
            with torch.no_grad():
                param_norm = torch.linalg.vector_norm(
                    torch.stack([p.norm() for p in model.parameters()])).item()
            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["steps"].append(step)
            history["tokens"].append(tokens_seen)
            print(f"[val step {step}] train loss {train_loss:.3f}, val loss {val_loss:.3f}, "
                  f"param norm {param_norm:.1f}")
            if metrics:
                metrics("val", step=step, tokens=tokens_seen, train_loss=train_loss,
                        val_loss=val_loss, param_norm=param_norm)
            train_loss_sum, train_loss_steps = 0.0, 0

        if step % sample_freq == 0 or step == max_steps:
            print_samples(model, tokenizer, device, amp)

        if eval_sets and (tokens_seen >= next_eval or step == max_steps):
            holdout_eval(model, eval_sets, settings, step, tokens_seen, metrics)
            while next_eval <= tokens_seen:
                next_eval += settings.eval_every_tokens

        if checkpointer and (stop or step == max_steps or checkpointer.due()):
            # On a stop, wait for the S3 copy too: the machine may be about to go away.
            checkpointer.save(make_state(step, tokens_seen, history), wait_upload=bool(stop) or step == max_steps)
        if stop:
            print(f"Stopped at step {step} ({stop.reason}). Resume with: "
                  f"uv run nanomix-train --resume {checkpointer.path if checkpointer else '<checkpoint>'}")
            return history, False

        # tok/s measures training only, not the val loss, sampling, evals and checkpoints above.
        log_start += time.time() - paused

    return history, True
