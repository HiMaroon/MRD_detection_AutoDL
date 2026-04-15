from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import csv
import math
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .augmentations import (
    BorderSuppressionAugment,
    CenterWeightTransform,
    HEDeconvolution,
    MacenkoNormalizer,
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
    bbox: Optional[Tuple[float, float, float, float]] = None
    mpp: Optional[float] = None


class LabelFileDataset(Dataset):
    """
    Backward-compatible dataset + extended modes:
    - image_only (default)
    - roi_size (single ROI + size features)
    - dual_scale_size (tight/context + size features)
    """

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
        self.input_mode = self.advanced_cfg.get("input_mode", "image_only")

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
        self.stain_norm = MacenkoNormalizer(
            enabled=bool(self.advanced_cfg.get("use_stain_normalization", False)),
        )
        self.he_deconv = HEDeconvolution(
            enabled=bool(self.advanced_cfg.get("use_he_channels", False)),
            output_mode=self.advanced_cfg.get("he_output_mode", "analysis"),
        )

        self.roi_expand_ratio = float(self.advanced_cfg.get("roi_expand_ratio", 0.25))
        self.context_expand_ratio = float(self.advanced_cfg.get("context_expand_ratio", 0.6))

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
                y = 0 if big_label == 2 else big_label
                records.append(SampleRecord(img_path=img, label=y))

        bbox_csv = self.advanced_cfg.get("bbox_csv", "")
        if bbox_csv:
            bbox_map = self._load_bbox_csv(bbox_csv)
            out = []
            for rec in records:
                key = rec.img_path
                if key in bbox_map:
                    b, mpp = bbox_map[key]
                    rec.bbox = b
                    rec.mpp = mpp
                out.append(rec)
            return out
        return records

    def _load_bbox_csv(self, bbox_csv: str):
        mapping = {}
        with open(bbox_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                mpp = float(row["mpp"]) if row.get("mpp") not in (None, "") else None
                mapping[img_path] = ((x1, y1, x2, y2), mpp)
        return mapping

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

    def _clip_box(self, box, w, h):
        x1, y1, x2, y2 = box
        return (
            max(0.0, min(float(w - 1), x1)),
            max(0.0, min(float(h - 1), y1)),
            max(0.0, min(float(w), x2)),
            max(0.0, min(float(h), y2)),
        )

    def _expand_box(self, box, w, h, ratio):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = max(1.0, (x2 - x1) * (1 + ratio))
        bh = max(1.0, (y2 - y1) * (1 + ratio))
        nx1 = cx - bw / 2
        ny1 = cy - bh / 2
        nx2 = cx + bw / 2
        ny2 = cy + bh / 2
        return self._clip_box((nx1, ny1, nx2, ny2), w, h)

    def _crop_with_padding(self, pil_img: Image.Image, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = pil_img.crop((x1, y1, x2, y2))
        canvas = Image.new("RGB", (self.img_size, self.img_size), color=(int(self.mean[0] * 255), int(self.mean[1] * 255), int(self.mean[2] * 255)))
        crop.thumbnail((self.img_size, self.img_size), resample=Image.BILINEAR)
        px = (self.img_size - crop.size[0]) // 2
        py = (self.img_size - crop.size[1]) // 2
        canvas.paste(crop, (px, py))
        return canvas

    def _size_features(self, box, mpp=None):
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area = w * h
        ar = w / h
        eqd = math.sqrt(4.0 * area / math.pi)

        feats = [w, h, area, ar, eqd]
        if mpp is not None and mpp > 0:
            wu = w * mpp
            hu = h * mpp
            area_u = area * mpp * mpp
            eqd_u = eqd * mpp
            feats.extend([wu, hu, area_u, eqd_u])

        feats = np.array(feats, dtype=np.float32)
        skew_idx = np.array([0, 1, 2, 4] + ([5, 6, 7, 8] if len(feats) > 5 else []))
        feats[skew_idx] = np.log1p(feats[skew_idx])
        return torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return self.base_len * self.repeat_factor

    def _post_process_tensor(self, x: torch.Tensor):
        x = self.stain_norm(x)
        x, he_info = self.he_deconv(x)
        if self.training:
            x = self.border_aug(x)
        x = self.center_weight_transform(x)
        x = self._normalize(x)
        return x, he_info

    def __getitem__(self, idx):
        if self.base_len == 0:
            raise IndexError("Dataset is empty.")

        rec = self.samples[idx % self.base_len]
        try:
            pil = Image.open(rec.img_path).convert("RGB")
            w, h = pil.size

            if self.input_mode == "image_only":
                img = self.transform(pil)
                img, he_info = self._post_process_tensor(img)
                out = {"image": img, "target": rec.label}
                if he_info is not None:
                    out["he"] = he_info
                return out

            box = rec.bbox if rec.bbox is not None else (0.0, 0.0, float(w), float(h))
            box = self._clip_box(box, w, h)
            tight_box = self._expand_box(box, w, h, self.roi_expand_ratio)
            roi_pil = self._crop_with_padding(pil, tight_box)
            roi_t = self.transform(roi_pil)
            roi_t, he_info = self._post_process_tensor(roi_t)
            size_feat = self._size_features(box, rec.mpp)

            if self.input_mode == "roi_size":
                out = {"image": roi_t, "size_features": size_feat, "target": rec.label}
                if he_info is not None:
                    out["he"] = he_info
                return out

            if self.input_mode == "dual_scale_size":
                context_box = self._expand_box(box, w, h, self.context_expand_ratio)
                context_pil = self._crop_with_padding(pil, context_box)
                context_t = self.transform(context_pil)
                context_t, _ = self._post_process_tensor(context_t)
                out = {
                    "image_tight": roi_t,
                    "image_context": context_t,
                    "size_features": size_feat,
                    "target": rec.label,
                }
                if he_info is not None:
                    out["he"] = he_info
                return out

            raise ValueError(f"unknown input_mode={self.input_mode}")

        except Exception as e:
            print(f"\n[ERROR] Failed to load: {rec.img_path} | Error: {e}")
            dummy = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            return {"image": dummy, "target": rec.label}
