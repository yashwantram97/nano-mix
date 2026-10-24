"""Training and validation batches from the OLMo-core mix that mix.py builds (settings in configs/mix.yaml).

The mix is built with the same code as mix.py and verify_mix.py, so training reads exactly the
instances verify_mix.py checked. A fixed, seeded set of instances is held out for validation; the
rest is read in a seeded random order, at most once, so nothing is repeated.
"""

import signal

import torch
from torch.utils.data import DataLoader, Dataset

from nanomix.data.mix import build_dataset, check_coverage, load_config


# Target id that cross_entropy skips (its default ignore_index).
IGNORE_INDEX = -100


class MixInstances(Dataset):
    """(input, target) pairs from mix instances, as OLMo-core builds them (data/utils.py get_labels).

    The input is the whole instance (sequence_length tokens, so shapes stay exactly 1024); the
    targets are the same tokens shifted left by one, with the last position set to IGNORE_INDEX
    because its next token isn't in the instance. sequence_length - 1 targets are scored.
    """

    def __init__(self, mix, indices):
        self.mix = mix
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ids = torch.as_tensor(self.mix[int(self.indices[i])]["input_ids"], dtype=torch.long)
        targets = torch.cat([ids[1:], ids.new_tensor([IGNORE_INDEX])])
        return ids, targets


def ignore_stop_signals(_worker_id=None):
    """DataLoader worker_init_fn: workers ignore Ctrl-C / SIGTERM / SIGHUP, so the main process
    can finish its step and save. (Workers exit on their own when the main process does.)"""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)


def create_mix_dataloaders(mix_config, batch_size, val_instances, num_workers=0, pin_memory=False,
                           skip_instances=0):
    """Returns (train_loader, val_loader, mix config dict).

    skip_instances drops that many from the start of the training order: a resumed run passes
    what it already trained on, so it continues with exactly the next batch.
    """
    cfg = load_config(mix_config)
    check_coverage(cfg)
    mix, _ = build_dataset(cfg)

    order = torch.randperm(len(mix), generator=torch.Generator().manual_seed(cfg["seed"]))
    val_idx, train_idx = order[:val_instances], order[val_instances + skip_instances:]

    def loader(indices, workers):
        # No shuffle here: the indices are already in a seeded random order. persistent_workers:
        # the workers (each holding a copy of the OLMo-core dataset) start once, not per epoch.
        return DataLoader(MixInstances(mix, indices), batch_size=batch_size, drop_last=True,
                          num_workers=workers, pin_memory=pin_memory,
                          persistent_workers=workers > 0,
                          worker_init_fn=ignore_stop_signals if workers else None)

    # The trainer reads a few val batches once, at the start; worker processes would only sit idle.
    return loader(train_idx, num_workers), loader(val_idx, 0), cfg
