"""Training and validation batches from the OLMo-core mix that mix.py builds (settings in mix.yaml).

The mix is built with the same code as mix.py and verify_mix.py, so training reads exactly the
instances verify_mix.py checked. A fixed, seeded set of instances is held out for validation; the
rest is read in a seeded random order, at most once, so nothing is repeated.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from mix import build_dataset, check_coverage, load_config


class MixInstances(Dataset):
    """(input, target) pairs from mix instances: an instance of sequence_length tokens gives sequence_length - 1 next-token targets."""

    def __init__(self, mix, indices):
        self.mix = mix
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ids = torch.as_tensor(self.mix[int(self.indices[i])]["input_ids"], dtype=torch.long)
        return ids[:-1], ids[1:]


def create_mix_dataloaders(mix_config, batch_size, val_instances, num_workers=0, pin_memory=False):
    """Returns (train_loader, val_loader, mix config dict)."""
    cfg = load_config(mix_config)
    check_coverage(cfg)
    mix, _ = build_dataset(cfg)

    order = torch.randperm(len(mix), generator=torch.Generator().manual_seed(cfg["seed"]))
    val_idx, train_idx = order[:val_instances], order[val_instances:]

    def loader(indices):
        # No shuffle here: the indices are already in a seeded random order.
        return DataLoader(MixInstances(mix, indices), batch_size=batch_size, drop_last=True,
                          num_workers=num_workers, pin_memory=pin_memory)

    return loader(train_idx), loader(val_idx), cfg
