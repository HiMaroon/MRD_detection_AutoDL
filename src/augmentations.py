import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
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


class MacenkoNormalizer:
    """Lightweight Macenko-style normalization using stain OD matrix projection."""

    def __init__(self, enabled: bool = False, alpha: float = 1.0, beta: float = 0.15):
        self.enabled = enabled
        self.alpha = alpha
        self.beta = beta
        self.he_ref = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=np.float32)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = _ensure_tensor_3c(img_tensor)
        if not self.enabled:
            return img_tensor

        rgb = img_tensor[:3].permute(1, 2, 0).cpu().numpy().clip(1e-6, 1.0)
        od = -np.log(rgb)
        flat = od.reshape(-1, 3)
        mask = np.all(flat > self.beta, axis=1)
        if mask.sum() < 10:
            return img_tensor

        od_f = flat[mask]
        cov = np.cov(od_f.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top2 = eigvecs[:, np.argsort(eigvals)[-2:]]
        proj = od_f @ top2
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        v1 = top2 @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = top2 @ np.array([np.cos(max_phi), np.sin(max_phi)])
        he = np.stack([v1 / (np.linalg.norm(v1) + 1e-8), v2 / (np.linalg.norm(v2) + 1e-8)], axis=1)

        y = np.linalg.lstsq(he, flat.T, rcond=None)[0]
        y = np.clip(y, 0.0, None)
        recon = np.exp(-(self.he_ref.T @ y)).T.reshape(rgb.shape)
        recon = np.clip(recon, 0.0, 1.0)

        out = img_tensor.clone()
        out[:3] = torch.from_numpy(recon).permute(2, 0, 1).to(img_tensor.dtype)
        return out


class HEDeconvolution:
    def __init__(self, enabled: bool = False, output_mode: str = "analysis"):
        self.enabled = enabled
        self.output_mode = output_mode
        self.he_matrix = torch.tensor(
            [[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]],
            dtype=torch.float32,
        )

    def __call__(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        img_tensor = _ensure_tensor_3c(img_tensor)
        if not self.enabled:
            return img_tensor, None

        rgb = img_tensor[:3].clamp(1e-6, 1.0)
        od = -torch.log(rgb).permute(1, 2, 0).reshape(-1, 3)
        he_inv = torch.inverse(self.he_matrix.to(rgb.device))
        stains = (od @ he_inv.T).reshape(rgb.shape[1], rgb.shape[2], 3).permute(2, 0, 1)
        h_ch = stains[0].clamp(min=0)
        e_ch = stains[1].clamp(min=0)

        info = {"H": h_ch, "E": e_ch}

        if self.output_mode == "extra_channels":
            he_stack = torch.stack([h_ch, e_ch], dim=0)
            out = torch.cat([img_tensor[:3], he_stack], dim=0)
            return out, info

        return img_tensor, info


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
