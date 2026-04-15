from collections import Counter

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import LabelFileDataset


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, cfg_data, batch_size, num_workers, pin_memory=True):
        super().__init__()
        self.cfg = cfg_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds = None
        self.val_ds = None
        self.class_weights = None
        self.train_sampler = None
        self.class_counts = None

    def setup(self, stage=None):
        advanced_cfg = self.cfg.get("advanced", {})
        self.train_ds = LabelFileDataset(
            self.cfg["train_labels"],
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            self.cfg.get("augment"),
            True,
            self.cfg.get("repeat_factor", 1),
            advanced_cfg=advanced_cfg,
        )
        self.val_ds = LabelFileDataset(
            self.cfg["val_labels"],
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            None,
            False,
            1,
            advanced_cfg=advanced_cfg,
        )

        counts = Counter([s.label for s in self.train_ds.samples])
        self.class_counts = counts
        total = sum(counts.values())

        self.class_weights = torch.tensor(
            [
                0.0 if counts.get(c, 0) == 0 else total / (self.cfg["num_classes"] * counts.get(c, 0))
                for c in range(self.cfg["num_classes"])
            ],
            dtype=torch.float,
        )

        if self.cfg.get("use_weighted_sampler", False):
            sample_weights = []
            for s in self.train_ds.samples:
                c = counts.get(s.label, 0)
                sample_weights.append(0.0 if c == 0 else 1.0 / c)

            self.train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(f"[Data] WeightedRandomSampler enabled. class_counts={dict(counts)}")
        else:
            self.train_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_exception(self, exception: BaseException):
        pass
