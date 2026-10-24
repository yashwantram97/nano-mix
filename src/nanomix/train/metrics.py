"""Where the numbers go: the JSONL metrics file, W&B and its alerts, MFU, and the loss plot."""

import json
import os
import time

import matplotlib.pyplot as plt
import torch

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


class MetricsLog:
    """Appends one JSON record per line to path and flushes, so a killed run keeps its metrics.

    With a W&B run, train / val / BPB records are also logged there (as train/loss, val/val_loss,
    bpb/bpb/fineweb_edu, ... against the optimizer step). The JSONL file stays the full record.
    """

    def __init__(self, path, wandb_run=None, run_id="run"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.file = open(path, "a")
        self.wandb_run = wandb_run

    def __call__(self, kind, **fields):
        self.file.write(json.dumps({"type": kind, "time": round(time.time(), 1), **fields}) + "\n")
        self.file.flush()
        if self.wandb_run is not None and kind in ("train", "val", "bpb"):
            # W&B drops points at a step below its latest, so after a resume the steps redone
            # since the checkpoint keep their first values.
            self.wandb_run.log({f"{kind}/{k}": v for k, v in fields.items()
                                if k != "step" and isinstance(v, (int, float))}, step=fields["step"])

    def alert(self, title, text):
        """Print, and send a W&B alert (email / Slack, as set in W&B's user settings), at most
        one per title every 30 min."""
        print(f"ALERT: {title}: {text}")
        if self.wandb_run is not None:
            import wandb
            self.wandb_run.alert(title=title, text=text, level=wandb.AlertLevel.WARN, wait_duration=1800)

    def finish(self):
        self.file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def init_wandb(settings, name, wandb_id, config):
    """A W&B run for this run id (resumed when wandb_id is given), or None if wandb_project is unset."""
    if not settings.wandb_project:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("wandb_project is set but WANDB_API_KEY isn't: put it in .env (or the "
                         "environment), or set wandb_project=None to log to the JSONL file only")
    import wandb
    return wandb.init(project=settings.wandb_project, entity=settings.wandb_entity, name=name,
                      id=wandb_id, resume="allow" if wandb_id else None, config=config)


def save_loss_plot(history, path):
    """Train and val loss against steps (bottom axis) and tokens seen (top axis), saved to path."""
    fig, ax1 = plt.subplots()
    ax1.plot(history["steps"], history["train_losses"], label="Training loss")
    ax1.plot(history["steps"], history["val_losses"], linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()  # a second x-axis sharing the y-axis
    ax2.plot(history["tokens"], history["train_losses"], alpha=0)  # invisible, only aligns the ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
