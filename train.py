# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import contextlib
import dataclasses
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer


# Import from local files
from config import MODEL_CONFIGS, TRAIN_PRESETS, model_label
from model import GPTModel, mask_padded_logits
from checkpointing import Checkpointer, StopRequest, check_resume, load_checkpoint, to_cpu
from dataloader import create_mix_dataloaders
from eval_bpb import BPB_WINDOW, evaluate_bpb, load_eval_sets

# Fixed prompts sampled (greedy, 100 tokens) at every holdout eval.
SAMPLE_PROMPTS = {
    "prose": "The history of the printing press",
    "code": "def fibonacci(n):\n",
    "math": "Solve for x: 3x + 7 = 22.\nSolution:",
}
SOURCE_ABBREV = {"fineweb_edu": "fw", "python_edu": "py", "finemath": "fm", "gutenberg": "gb"}

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # Padded vocab rows decode to nothing, so they must never be selected.
        logits = mask_padded_logits(logits[:, -1, :], model.vocab_size)

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(mix_cfg, gpt_config):
    """e.g. fw60_py20_fm13_gb07_s1234_124m: mix ratios in percent, the mix seed, the model size."""
    parts = [f"{SOURCE_ABBREV.get(n, n)}{round(r * 100):02d}" for n, r in mix_cfg["sources"].items()]
    return "_".join(parts) + f"_s{mix_cfg['seed']}_{model_label(gpt_config)}"


def resolve_precision(precision, device):
    """Downgrade to fp32 where bf16 is not actually supported.

    Pre-Ampere CUDA cards have no bf16 path; silently running there would be
    slower than fp32, not faster.
    """
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU (pre-Ampere); falling back to fp32")
        return "fp32"
    return precision


def autocast_ctx(device, precision):
    """Context manager wrapping forward + loss.

    bf16 carries fp32's exponent range, so unlike fp16 there are no gradient
    underflow problems and no GradScaler is needed. Backward is deliberately
    left outside: autocast covers the forward ops, and autograd then runs each
    backward in the dtype its forward was recorded in.
    """
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)  # as in tokenize_data.py
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device, amp=None):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    with (amp if amp is not None else contextlib.nullcontext()):
        logits = model(input_batch)
        # cross_entropy upcasts to fp32 internally under autocast, so the
        # softmax and its gradient are not computed in bf16. Targets equal to
        # -100 (the last position of each instance, dataloader.IGNORE_INDEX) are
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
        total = sum(calc_loss_batch(x, y, model, device, amp) for x, y in val_batches)
    model.train()
    return (total / len(val_batches)).item()


# Dense bf16 peak in TFLOP/s, matched against torch.cuda.get_device_name() in this order.
# p5 instances have the SXM H100 (989); the PCIe card is 756.
PEAK_TFLOPS = [("H100 PCIe", 756), ("H100 NVL", 835), ("H100", 989), ("H200", 989), ("A100", 312)]


def resolve_peak_flops(device, settings):
    """GPU peak in FLOP/s for MFU, or None if unknown (MFU isn't printed then)."""
    if settings.peak_tflops:
        return settings.peak_tflops * 1e12
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return next((tflops * 1e12 for key, tflops in PEAK_TFLOPS if key in name), None)


def training_flops_per_token(cfg, n_params):
    """6N for the matmuls (forward + backward; the tied embedding counts once, as the output
    head) plus 12 * layers * width * context for attention scores, as in PaLM / nanoGPT."""
    return 6 * n_params + 12 * cfg.n_layers * cfg.emb_dim * cfg.context_length


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


class MetricsLog:
    """Appends one JSON record per line to path and flushes, so a killed run keeps its metrics."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a")

    def __call__(self, kind, **fields):
        self.file.write(json.dumps({"type": kind, "time": round(time.time(), 1), **fields}) + "\n")
        self.file.flush()


def generate_and_print_sample(model, tokenizer, device, start_context, amp=None):
    model.eval()
    context_size = model.context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad(), (amp if amp is not None else contextlib.nullcontext()):
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def print_holdout_eval(model, eval_sets, tokenizer, device, settings, step, tokens_seen, amp=None,
                       metrics=None):
    """Holdout BPB per eval set, then greedy samples from the fixed prompts."""
    t0 = time.time()
    bpb = evaluate_bpb(model, eval_sets, settings.batch_size)
    print(f"[eval {tokens_seen:,} tokens] " + "  ".join(f"bpb/{n} {v:.4f}" for n, v in bpb.items())
          + f"  ({time.time() - t0:.0f}s)")
    if metrics:
        metrics("bpb", step=step, tokens=tokens_seen, **{f"bpb/{n}": v for n, v in bpb.items()})
    model.eval()
    for name, prompt in SAMPLE_PROMPTS.items():
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        with torch.no_grad(), (amp if amp is not None else contextlib.nullcontext()):
            out = generate_text_simple(model, encoded, max_new_tokens=100,
                                       context_size=model.context_length)
        print(f"[sample/{name}] {token_ids_to_text(out, tokenizer)!r}")
    model.train()


def train_model_simple(model, train_model, train_loader, val_batches, optimizer, device, settings,
                       start_context, tokenizer, amp=None, eval_sets=None, flops_per_token=None,
                       peak_flops=None, metrics=None, start_step=0, tokens_seen=0, history=None,
                       checkpointer=None, make_state=None, stop=None):
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
    (train_losses, val_losses, tokens seen, steps, finished).
    """
    history = history or {"train_losses": [], "val_losses": [], "steps": [], "tokens": []}
    train_losses, val_losses = history["train_losses"], history["val_losses"]
    track_steps, track_tokens_seen = history["steps"], history["tokens"]

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

    train_model.train()
    # One pass: main() checks the mix holds max_steps of batches, so this never
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
            loss = calc_loss_batch(input_batch, target_batch, train_model, device, amp)
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
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_steps.append(step)
            track_tokens_seen.append(tokens_seen)
            print(f"[val step {step}] train loss {train_loss:.3f}, val loss {val_loss:.3f}, "
                  f"param norm {param_norm:.1f}")
            if metrics:
                metrics("val", step=step, tokens=tokens_seen, train_loss=train_loss,
                        val_loss=val_loss, param_norm=param_norm)
            train_loss_sum, train_loss_steps = 0.0, 0

        if step % sample_freq == 0 or step == max_steps:
            generate_and_print_sample(model, tokenizer, device, start_context, amp)

        if eval_sets and (tokens_seen >= next_eval or step == max_steps):
            print_holdout_eval(model, eval_sets, tokenizer, device, settings, step, tokens_seen, amp,
                               metrics)
            while next_eval <= tokens_seen:
                next_eval += settings.eval_every_tokens

        if checkpointer and (stop or step == max_steps or checkpointer.due()):
            # On a stop, wait for the S3 copy too: the machine may be about to go away.
            checkpointer.save(make_state(step, tokens_seen, history), wait_upload=bool(stop) or step == max_steps)
        if stop:
            print(f"Stopped at step {step} ({stop.reason}). Resume with: "
                  f"uv run python train.py --resume {checkpointer.path if checkpointer else '<checkpoint>'}")
            return train_losses, val_losses, track_tokens_seen, track_steps, False

        # tok/s measures training only, not the val loss, sampling, evals and checkpoints above.
        log_start += time.time() - paused

    return train_losses, val_losses, track_tokens_seen, track_steps, True


def plot_losses(steps_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against steps
    ax1.plot(steps_seen, train_losses, label="Training loss")
    ax1.plot(steps_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings, resume=None, tag=None):
    """Train a new run, or continue the one in the checkpoint at ``resume``.

    The run id is the run name (+ "_<tag>"); checkpoints go to <checkpoint_dir>/<run_id>/ and
    metrics to <metrics_dir>/<run_id>.jsonl. A new run refuses to start over an existing one.
    """

    torch.manual_seed(123)
    device = resolve_device()

    precision = resolve_precision(settings.precision, device)
    amp = autocast_ctx(device, precision)
    print(f"Using device: {device} ({precision})")

    if device.type == "cuda":
        # TF32 for the fp32 ops autocast leaves alone; negligible accuracy
        # cost at this scale, meaningful speedup on the matmuls.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ##############################
    # Initialize model
    ##############################

    ckpt = load_checkpoint(resume) if resume else None
    if ckpt and ckpt["model_config"] != dataclasses.asdict(gpt_config):
        raise SystemExit(f"Can't resume: {resume} was trained with a different model config (--model)")
    model = GPTModel(gpt_config)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
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

    ##############################
    # Set up dataloaders
    ##############################

    # The OLMo-core mix from mix.yaml, built the same way as mix.py / verify_mix.py. A resumed
    # run skips the instances it already trained on (the order is a fixed seeded permutation).
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

    ##############################
    # Train model
    ##############################

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

    metrics = MetricsLog(metrics_path)
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
        }

    train_losses, val_losses, tokens_seen, steps_seen, finished = train_model_simple(
        model, train_model, train_loader, val_batches, optimizer, device, settings,
        start_context="Every effort moves you", tokenizer=tokenizer, amp=amp,
        eval_sets=eval_sets, flops_per_token=flops_per_token, peak_flops=peak_flops,
        metrics=metrics, start_step=start_step,
        tokens_seen=ckpt["tokens_seen"] if ckpt else 0, history=ckpt["history"] if ckpt else None,
        checkpointer=checkpointer, make_state=make_state, stop=StopRequest(settings.watch_spot),
    )
    if finished:
        print(f"Done. Final checkpoint: {checkpointer.path} "
              f"(weights: torch.load(path, weights_only=True)['model'])")

    return train_losses, val_losses, tokens_seen, steps_seen, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="124m", help="config.py MODEL_CONFIGS name")
    parser.add_argument("--preset", choices=TRAIN_PRESETS, default="mac",
                        help="config.py TRAIN_PRESETS name: mac (quick test) or h100 (the 5B run)")
    parser.add_argument("--max-steps", type=int, help="override the preset's max_steps (smoke runs)")
    parser.add_argument("--tag", help="suffix for the run id, e.g. smoke: a separate run with its own "
                                      "checkpoints and metrics")
    parser.add_argument("--resume", help="continue from this checkpoint (checkpoints/<run_id>/ckpt.pt); "
                                         "--model, --preset and --max-steps must match it")
    args = parser.parse_args()
    model_config = MODEL_CONFIGS[args.model]
    settings = TRAIN_PRESETS[args.preset]
    if args.max_steps:
        settings = settings.replace(max_steps=args.max_steps)

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, steps_seen, model = main(model_config, settings, resume=args.resume, tag=args.tag)

    ###########################
    # After training
    ###########################

    # Plot results
    plot_losses(steps_seen, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")
