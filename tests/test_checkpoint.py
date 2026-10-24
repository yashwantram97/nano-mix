import dataclasses
import signal
import subprocess
import sys

import pytest
import torch

from nanomix.data.mix import MIX_KEYS
from nanomix.model.gpt import GPTModel
from nanomix.train.checkpoint import Checkpointer, check_resume, load_checkpoint, to_cpu
from nanomix.train.settings import MAC_SETTINGS

MIX = {"requested_tokens": 1000, "sequence_length": 1024, "global_batch_size": 8192, "seed": 1234,
       "tokenizer": "HuggingFaceTB/SmolLM2-135M", "sources": {"fineweb_edu": 1.0}}


def make_ckpt(model, cfg, settings=MAC_SETTINGS, mix=MIX):
    return {"run_id": "test", "step": 7, "tokens_seen": 700, "model": to_cpu(model.state_dict()),
            "rng": {"torch": torch.get_rng_state(), "cuda": None},
            "history": {"train_losses": [1.0], "val_losses": [2.0], "steps": [5], "tokens": [500]},
            "model_config": dataclasses.asdict(cfg), "train_settings": dataclasses.asdict(settings),
            "mix": mix, "wandb_id": None}


def test_save_rotate_and_load(tmp_path, tiny_config):
    model = GPTModel(tiny_config)
    checkpointer = Checkpointer(tmp_path / "run", minutes=None)
    checkpointer.save(make_ckpt(model, tiny_config))
    checkpointer.save({**make_ckpt(model, tiny_config), "step": 8})

    assert load_checkpoint(tmp_path / "run" / "ckpt_prev.pt")["step"] == 7
    ckpt = load_checkpoint(checkpointer.path)  # weights_only=True: only tensors and plain values
    assert ckpt["step"] == 8 and ckpt["history"]["steps"] == [5]
    restored = GPTModel(tiny_config)
    restored.load_state_dict(ckpt["model"])
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values()))
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_check_resume_accepts_safe_changes(tiny_config):
    ckpt = make_ckpt(GPTModel(tiny_config), tiny_config)
    safe = MAC_SETTINGS.replace(log_freq=1, eval_every_tokens=1, num_workers=0, mix_config="other.yaml")
    check_resume(ckpt, tiny_config, safe, MIX)


@pytest.mark.parametrize("change", [
    {"settings": MAC_SETTINGS.replace(learning_rate=1e-3)},
    {"settings": MAC_SETTINGS.replace(batch_size=4)},
    {"mix": {**MIX, "seed": 1}},
    {"mix": {**MIX, "sources": {"fineweb_edu": 0.5, "python_edu": 0.5}}},
    {"model": "other"},
])
def test_check_resume_refuses_changes(tiny_config, change):
    ckpt = make_ckpt(GPTModel(tiny_config), tiny_config)
    cfg = tiny_config.replace(n_layers=3) if "model" in change else tiny_config
    with pytest.raises(SystemExit, match="Can't resume"):
        check_resume(ckpt, cfg, change.get("settings", MAC_SETTINGS), change.get("mix", MIX))


def test_mix_keys_decide_the_instances():
    assert set(MIX_KEYS) == set(MIX)


# A stand-in for the trainer: line-buffered stdout as in train/cli.py, a StopRequest, and a save
# once it is set.
STOP_SCRIPT = """
import sys, time, torch
from nanomix.train.checkpoint import Checkpointer, StopRequest
sys.stdout.reconfigure(line_buffering=True)
stop = StopRequest(watch_spot=False)
checkpointer = Checkpointer(sys.argv[1], minutes=None)
print("ready")
while not stop:
    time.sleep(0.01)
    print("step")
checkpointer.save({"step": 1, "weights": torch.zeros(3)})
"""


def run_until_signal(tmp_path, sig, close_stdout):
    child = subprocess.Popen([sys.executable, "-c", STOP_SCRIPT, str(tmp_path / "run")],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert child.stdout.readline() == "ready\n"
    if close_stdout:  # like tee exiting on Ctrl-C: the trainer's next print hits a closed pipe
        child.stdout.close()
    child.send_signal(sig)
    child.wait(timeout=60)
    return (tmp_path / "run" / "ckpt.pt").exists()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_stop_signal_saves_a_checkpoint(tmp_path, sig):
    assert run_until_signal(tmp_path, sig, close_stdout=False)


@pytest.mark.xfail(strict=True, reason="open: the trainer dies on a closed stdout before saving "
                                        "(docs/roadmap.md section 3, 'writes its own console log')")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_stop_signal_saves_with_stdout_closed(tmp_path, sig):
    assert run_until_signal(tmp_path, sig, close_stdout=True)
