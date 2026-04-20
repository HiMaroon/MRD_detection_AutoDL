from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .augmentations import (
    BorderSuppressionAugment,
    CenterWeightTransform,
)


class DiscreteRotate:
    def __init__(self, angles):
        self.angles = [float(a) for a in angles]

    def __call__(self, img):
        return TF.rotate(img, random.choice(self.angles))


@dataclass
class SampleRecord:
    img_path: str
    label: int


class LabelFileDataset(Dataset):
    """Image-only dataset for baseline / center-border experiments."""

    def __init__(
        self,
        label_file,
        img_size,
        mean,
        std,
        augment=None,
        training=True,
        repeat_factor=1,
        advanced_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.advanced_cfg = advanced_cfg or {}
        self.samples = self._load_label_file(label_file)
        self.img_size = int(img_size)
        self.mean = mean
        self.std = std
        self.training = training
        self.repeat_factor = max(1, int(repeat_factor)) if training else 1
        self.base_len = len(self.samples)

        self.center_weight_transform = CenterWeightTransform(
            mode=self.advanced_cfg.get("center_weight_mode", "none"),
            strength=float(self.advanced_cfg.get("center_weight_strength", 0.0)),
            sigma=float(self.advanced_cfg.get("center_weight_sigma", 0.45)),
        )
        self.border_aug = BorderSuppressionAugment(
            prob=float(self.advanced_cfg.get("border_aug_prob", 0.0)) if training else 0.0,
            width_ratio=float(self.advanced_cfg.get("border_aug_width_ratio", 0.12)),
            mode=self.advanced_cfg.get("border_aug_mode", "blur"),
        )
        self.transform = self._build_transform(img_size, mean, std, augment, training)

    def _load_label_file(self, label_file: str) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.rsplit(maxsplit=2)
                if len(parts) != 3:
                    continue
                img = parts[0]
                big_label = int(parts[1])
                small_label = int(parts[2])
                # y = 0 if big_label == 2 else big_label
                y = small_label
                records.append(SampleRecord(img_path=img, label=y))

        return records

    def _build_transform(self, img_size, mean, std, augment, training):
        t = []

        if training and augment:
            random_crop_scale = augment.get("random_crop_scale", None)
            if random_crop_scale is not None:
                t.append(T.RandomResizedCrop(size=img_size, scale=tuple(random_crop_scale), ratio=(0.9, 1.1)))
            else:
                t.append(T.Resize((img_size, img_size)))

            if augment.get("hflip_p", 0.0) > 0:
                t.append(T.RandomHorizontalFlip(p=augment.get("hflip_p", 0.0)))
            if augment.get("vflip_p", 0.0) > 0:
                t.append(T.RandomVerticalFlip(p=augment.get("vflip_p", 0.0)))

            rotate_angles = augment.get("rotate_angles", None)
            if rotate_angles:
                t.append(DiscreteRotate(rotate_angles))
            elif augment.get("rotate_deg", 0) > 0:
                t.append(T.RandomRotation(degrees=augment.get("rotate_deg", 0)))

            if any(k in augment for k in ["brightness", "contrast", "saturation", "hue"]):
                t.append(
                    T.ColorJitter(
                        brightness=augment.get("brightness", 0),
                        contrast=augment.get("contrast", 0),
                        saturation=augment.get("saturation", 0),
                        hue=augment.get("hue", 0),
                    )
                )

            sharpness_factor = augment.get("sharpness_factor", None)
            sharpness_p = augment.get("sharpness_p", 0.0)
            if sharpness_factor is not None and sharpness_p > 0:
                t.append(T.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_p))
        else:
            t.append(T.Resize((img_size, img_size)))

        t.extend([T.ToTensor()])
        return T.Compose(t)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] >= 3:
            x[:3] = T.Normalize(mean=self.mean, std=self.std)(x[:3])
        return x


    def __len__(self):
        return self.base_len * self.repeat_factor

    def _post_process_tensor(self, x: torch.Tensor):
        if self.training:
            x = self.border_aug(x)
        x = self.center_weight_transform(x)
        x = self._normalize(x)
        return x

    def __getitem__(self, idx):
        if self.base_len == 0:
            raise IndexError("Dataset is empty.")

        rec = self.samples[idx % self.base_len]
        try:
            pil = Image.open(rec.img_path).convert("RGB")
            img = self.transform(pil)
            img = self._post_process_tensor(img)
            return {"image": img, "target": rec.label}

        except Exception as e:
            print(f"\n[ERROR] Failed to load: {rec.img_path} | Error: {e}")
            dummy = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            return {"image": dummy, "target": rec.label}
