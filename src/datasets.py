from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import csv
import os

from PIL import Image
import numpy as np
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
        return_morph: bool = False,
        morph_csv_path: Optional[str] = None,
        morph_cols: Optional[List[str]] = None,
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

        self.return_morph = bool(return_morph)
        self.morph_cols = list(morph_cols or ["area", "perimeter", "circularity"])
        self.morph_dim = len(self.morph_cols)
        self.morph_dict_by_path: Dict[str, np.ndarray] = {}
        self.morph_dict_by_filename: Dict[str, np.ndarray] = {}
        self.morph_dict_by_stem: Dict[str, np.ndarray] = {}
        if self.return_morph:
            self._init_morph_table(morph_csv_path, self.morph_cols)

    def _load_label_file(self, label_file: str) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        label_task = str(self.advanced_cfg.get("label_task", "")).strip().lower()
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
                # if small_label == 0:
                #     continue
                if label_task == "nm_three_class":
                    # 3-class NM screening task: 0 = other/background/negative, 1 = N, 2 = M.
                    # The label generator emits N=1, M=2, known negatives=3, unknown/background=0.
                    y = big_label if big_label in (1, 2) else 0
                else:
                    y=big_label
                    if big_label == 3:
                        y = 0
                    elif big_label == 2:
                        y = 1
                # y = 0 if small_label == 15 else small_label
                # y = small_label
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

    def _init_morph_table(self, morph_csv_path: Optional[str], morph_cols: List[str]):
        if not morph_csv_path or (not os.path.exists(morph_csv_path)):
            return

        with open(morph_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_path = (
                    row.get("image_path")
                    or row.get("img_path")
                    or row.get("path")
                    or ""
                ).strip()
                filename = (row.get("filename") or "").strip()

                vals: List[float] = []
                try:
                    for col in morph_cols:
                        value = float(row[col])
                        if col in ["area", "perimeter", "post_crop_area", "post_crop_perimeter", "pre_crop_area", "pre_crop_perimeter"]:
                            value = float(np.log1p(max(value, 0.0)))
                        vals.append(value)
                except (TypeError, ValueError, KeyError):
                    continue

                arr = np.asarray(vals, dtype=np.float32)

                if raw_path:
                    norm_path = os.path.normpath(raw_path)
                    self.morph_dict_by_path[norm_path] = arr
                    base_name = os.path.basename(norm_path)
                    if base_name:
                        self.morph_dict_by_filename[base_name] = arr

                if filename:
                    norm_file = os.path.basename(filename)
                    if norm_file:
                        self.morph_dict_by_filename[norm_file] = arr
                    stem = os.path.splitext(norm_file)[0]
                    if stem:
                        self.morph_dict_by_stem[stem] = arr
                elif raw_path:
                    stem = os.path.splitext(os.path.basename(raw_path))[0]
                    if stem:
                        self.morph_dict_by_stem[stem] = arr

    def _get_morph_label(self, path: str):
        norm_path = os.path.normpath(path)
        filename = os.path.basename(norm_path)
        stem = os.path.splitext(filename)[0]

        if norm_path in self.morph_dict_by_path:
            morph = self.morph_dict_by_path[norm_path]
            valid = np.ones(self.morph_dim, dtype=np.float32)
            return morph, valid

        if filename in self.morph_dict_by_filename:
            morph = self.morph_dict_by_filename[filename]
            valid = np.ones(self.morph_dim, dtype=np.float32)
            return morph, valid

        if stem in self.morph_dict_by_stem:
            morph = self.morph_dict_by_stem[stem]
            valid = np.ones(self.morph_dim, dtype=np.float32)
            return morph, valid

        morph = np.zeros(self.morph_dim, dtype=np.float32)
        valid = np.zeros(self.morph_dim, dtype=np.float32)
        return morph, valid

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
            out = {"image": img, "target": rec.label}
            if self.return_morph:
                morph, valid = self._get_morph_label(rec.img_path)
                out["morph"] = torch.from_numpy(morph)
                out["morph_valid"] = torch.from_numpy(valid)
            return out

        except Exception as e:
            print(f"\n[ERROR] Failed to load: {rec.img_path} | Error: {e}")
            dummy = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            out = {"image": dummy, "target": rec.label}
            if self.return_morph:
                out["morph"] = torch.zeros(self.morph_dim, dtype=torch.float32)
                out["morph_valid"] = torch.zeros(self.morph_dim, dtype=torch.float32)
            return out
