import math
import random
from typing import Optional

import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


def _ensure_tensor_3c(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"expect CHW tensor, got shape={tuple(img.shape)}")
    if img.shape[0] not in (3, 5):
        raise ValueError(f"expect 3 or 5 channels tensor, got shape={tuple(img.shape)}")
    return img


def _build_center_weight(h: int, w: int, mode: str, sigma: float, strength: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )

    if mode == "gaussian":
        dist2 = xx * xx + yy * yy
        base = torch.exp(-dist2 / max(1e-6, 2.0 * sigma * sigma))
    elif mode == "elliptical":
        sigma_x = max(1e-6, sigma)
        sigma_y = max(1e-6, sigma * 0.7)
        dist2 = (xx / sigma_x) ** 2 + (yy / sigma_y) ** 2
        base = torch.exp(-0.5 * dist2)
    else:
        return torch.ones(h, w)

    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    weight = 1.0 + strength * (base - 0.5)
    return weight.clamp(min=0.2)


class CenterWeightTransform:
    """Apply soft center prior without hard crop."""

    def __init__(self, mode: str = "none", strength: float = 0.0, sigma: float = 0.45):
        self.mode = mode
        self.strength = float(strength)
        self.sigma = float(sigma)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = _ensure_tensor_3c(img_tensor)
        if self.mode == "none" or self.strength <= 0:
            return img_tensor

        h, w = img_tensor.shape[-2:]
        weight = _build_center_weight(h, w, self.mode, self.sigma, self.strength).to(img_tensor.device)
        return img_tensor * weight.unsqueeze(0)


class BorderSuppressionAugment:
    """Augment only border area; keep central region unchanged."""

    def __init__(self, prob: float = 0.0, width_ratio: float = 0.12, mode: str = "blur"):
        self.prob = float(prob)
        self.width_ratio = float(width_ratio)
        self.mode = mode

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = _ensure_tensor_3c(img_tensor)
        if self.prob <= 0 or random.random() > self.prob:
            return img_tensor

        c, h, w = img_tensor.shape
        bw = max(1, int(min(h, w) * self.width_ratio))
        border_mask = torch.zeros((h, w), dtype=torch.bool, device=img_tensor.device)
        border_mask[:bw, :] = True
        border_mask[-bw:, :] = True
        border_mask[:, :bw] = True
        border_mask[:, -bw:] = True

        out = img_tensor.clone()

        if self.mode == "blur":
            pil = TF.to_pil_image(img_tensor[:3].cpu())
            blurred = TF.to_tensor(pil.filter(ImageFilter.GaussianBlur(radius=2.0))).to(img_tensor.device)
            out[:3, border_mask] = blurred[:, border_mask]
        elif self.mode == "low_contrast":
            mean = out[:, border_mask].mean(dim=1, keepdim=True)
            out[:, border_mask] = mean + 0.25 * (out[:, border_mask] - mean)
        elif self.mode == "random_mask":
            noise = torch.rand((c, border_mask.sum()), device=img_tensor.device)
            out[:, border_mask] = noise
        elif self.mode == "mean_color":
            mean_rgb = out[:, ~border_mask].mean(dim=1, keepdim=True)
            out[:, border_mask] = mean_rgb
        return out.clamp(0.0, 1.0)


def build_center_border_masks(h: int, w: int, border_ratio: float = 0.15, device: Optional[torch.device] = None):
    bh = max(1, int(h * border_ratio))
    bw = max(1, int(w * border_ratio))

    border = torch.zeros((h, w), dtype=torch.float32, device=device)
    border[:bh, :] = 1.0
    border[-bh:, :] = 1.0
    border[:, :bw] = 1.0
    border[:, -bw:] = 1.0
    center = 1.0 - border
    return border, center
