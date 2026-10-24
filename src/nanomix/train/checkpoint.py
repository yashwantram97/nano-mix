"""Training checkpoints: save, rotate, copy to S3, resume, and stop cleanly on a signal.

A checkpoint (checkpoints/<run_id>/ckpt.pt) holds everything needed to continue a run exactly:
model weights, AdamW state, step, tokens seen, RNG states, the loss history for the plot, and
the model / train / mix settings it was made with. All tensors are saved on CPU and every value is
a plain type, so torch.load(..., weights_only=True) reads it on any machine (a Mac included).

    ckpt.pt        latest      (written to ckpt.pt.tmp, then renamed: a crash mid-save can't
    ckpt_prev.pt   the one before    leave a half-written ckpt.pt)

The weights alone, e.g. for export: torch.load(path, map_location="cpu", weights_only=True)["model"].
eval/export_hf.py and eval/bpb.py accept a checkpoint directly (load_model_weights).

Checkpoints hold no Python module paths, so modules can be moved or renamed and old checkpoints
still load.
"""

import dataclasses
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import torch

from nanomix.data.mix import MIX_KEYS
from nanomix.model.config import ModelConfig

# TrainSettings fields that may differ on --resume: they don't change which batches are read or
# how the weights are updated, only logging, saving and speed. mix_config is only the file's
# path; the mix itself is compared through MIX_KEYS.
RESUME_SAFE = {"log_freq", "eval_freq", "eval_iter", "eval_every_tokens", "sample_freq", "compile",
               "num_workers", "peak_tflops", "metrics_dir", "checkpoint_minutes", "checkpoint_dir",
               "checkpoint_s3", "watch_spot", "wandb_project", "wandb_entity", "mix_config"}


def to_cpu(obj):
    """A copy of a (nested) state dict with every tensor moved to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_weights(path):
    """(state_dict, ModelConfig or None) from a training checkpoint or a weights-only .pth.

    A training checkpoint carries its own model config; a plain state dict doesn't.
    """
    obj = load_checkpoint(path)
    if "model" in obj and "optimizer" in obj:
        return obj["model"], ModelConfig(**obj["model_config"])
    return obj, None


def check_resume(ckpt, gpt_config, settings, mix_cfg):
    """SystemExit listing every setting that differs from the checkpoint's, except RESUME_SAFE."""
    problems = []
    if ckpt["model_config"] != dataclasses.asdict(gpt_config):
        problems.append("model config differs (--model)")
    saved = ckpt["train_settings"]
    for key, value in dataclasses.asdict(settings).items():
        if key not in RESUME_SAFE and saved.get(key) != value:
            problems.append(f"train setting {key}: checkpoint {saved.get(key)!r}, now {value!r}")
    for key in MIX_KEYS:
        if ckpt["mix"].get(key) != mix_cfg.get(key):
            problems.append(f"mix.yaml {key}: checkpoint {ckpt['mix'].get(key)!r}, now {mix_cfg.get(key)!r}")
    if problems:
        raise SystemExit("Can't resume: the run would not continue the same way.\n  " + "\n  ".join(problems))


class StopRequest:
    """Set by Ctrl-C, SIGTERM/SIGHUP (shutdown, tmux killed) or an EC2 spot interruption notice.

    The training loop checks it after each optimizer step, saves a checkpoint and returns. A
    second Ctrl-C stops at once, without saving.
    """

    def __init__(self, watch_spot=True):
        self.reason = None
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._handle)
        if watch_spot:
            threading.Thread(target=self._watch_spot, daemon=True).start()

    def __bool__(self):
        return self.reason is not None

    def _handle(self, signum, frame):
        if self.reason is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.reason = signal.Signals(signum).name
        print(f"\n{self.reason}: saving a checkpoint after this step (Ctrl-C again to quit without saving)")

    def _watch_spot(self):
        """Poll EC2 instance metadata (IMDSv2) every 5 s for a spot interruption notice, which
        comes ~2 minutes before the instance is stopped. Off EC2 the first request fails and
        the thread ends."""
        base = "http://169.254.169.254/latest"
        while self.reason is None:
            try:
                token = urllib.request.urlopen(urllib.request.Request(
                    f"{base}/api/token", method="PUT",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=1).read().decode()
            except OSError:
                return  # not on EC2 (or IMDS disabled)
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"{base}/meta-data/spot/instance-action",
                    headers={"X-aws-ec2-metadata-token": token}), timeout=1)
                self.reason = "spot interruption notice"
                print(f"\n{self.reason}: saving a checkpoint after this step")
                return
            except OSError:
                pass  # 404: no interruption scheduled
            time.sleep(5)


class Checkpointer:
    """Saves checkpoints for one run every `minutes`, keeps the last two, copies to S3."""

    def __init__(self, run_dir, minutes, s3=None, extra_uploads=()):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ckpt.pt"
        self.interval = minutes * 60 if minutes else None
        self.last = time.time()
        self.s3 = s3.rstrip("/") + "/" + self.dir.name if s3 else None
        self.extra_uploads = list(extra_uploads)  # e.g. the metrics .jsonl
        self.upload = None
        if self.s3 and not shutil.which("aws"):
            raise SystemExit(f"checkpoint_s3 is {s3} but the aws CLI isn't installed")

    def due(self):
        return self.interval is not None and time.time() - self.last >= self.interval

    def save(self, state, wait_upload=False):
        t0 = time.time()
        self.wait()  # never rotate a file that is still being uploaded
        tmp = self.path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        if self.path.exists():
            os.replace(self.path, self.dir / "ckpt_prev.pt")
        os.replace(tmp, self.path)
        self.last = time.time()
        print(f"[checkpoint] step {state['step']} -> {self.path} "
              f"({self.path.stat().st_size / 1e9:.2f} GB, {self.last - t0:.0f}s)")
        if self.s3:
            # In the background so training doesn't wait. An interrupted upload leaves the
            # previous S3 copy in place (S3 only commits a finished object).
            files = [self.path, *self.extra_uploads]
            self.upload = threading.Thread(target=self._upload, args=(files,), daemon=True)
            self.upload.start()
            if wait_upload:
                self.wait()

    def _upload(self, files):
        for f in files:
            result = subprocess.run(["aws", "s3", "cp", str(f), f"{self.s3}/{Path(f).name}",
                                     "--only-show-errors"], capture_output=True, text=True)
            if result.returncode:
                print(f"[checkpoint] WARNING: upload of {f} to {self.s3} failed: {result.stderr.strip()}")
        print(f"[checkpoint] copied to {self.s3}/")

    def wait(self):
        if self.upload is not None:
            self.upload.join()
            self.upload = None
