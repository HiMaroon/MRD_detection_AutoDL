import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .augmentations import build_center_border_masks


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=2, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.smoothing = smoothing

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.ones(num_classes, dtype=torch.float) * alpha
        else:
            raise TypeError(f"Unsupported alpha type: {type(alpha)}")

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        if self.smoothing > 0:
            targets_smooth = targets_onehot * (1 - self.smoothing) + self.smoothing / self.num_classes
        else:
            targets_smooth = targets_onehot

        pt = (probs * targets_onehot).sum(dim=1).clamp(min=1e-7, max=1 - 1e-7)
        ce_loss = -torch.sum(targets_smooth * torch.log(probs + 1e-7), dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0

        loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_norm = (x - mu) / sig

        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_norm * sig_mix + mu_mix


class ImageEncoder(nn.Module):
    def __init__(self, arch: str, pretrained: bool, drop: float, drop_path: float):
        super().__init__()
        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop,
            drop_path_rate=drop_path,
            in_chans=3,
        )
        self.num_features = getattr(self.backbone, "num_features", 1280)

    def forward(self, x):
        return self.backbone(x)


class HEBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.out_dim = 32

    def forward(self, he):
        return self.net(he)


class SizeMLP(nn.Module):
    def __init__(self, in_dim=5, hidden=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class LitSingleCell(nn.Module):
    def __init__(self, cfg_model, num_classes, class_weights=None, data_advanced_cfg=None):
        super().__init__()
        self.cfg_model = cfg_model
        self.num_classes = num_classes
        self.advanced_cfg = data_advanced_cfg or {}

        arch = cfg_model["arch"]
        pretrained = cfg_model.get("pretrained", True)
        drop = cfg_model.get("drop", 0.0)
        drop_path = cfg_model.get("drop_path", 0.0)

        self.image_encoder = ImageEncoder(arch, pretrained, drop, drop_path)
        self.mixstyle = MixStyle(p=cfg_model.get("mixstyle_prob", 0.0), alpha=cfg_model.get("mixstyle_alpha", 0.1))

        self.use_size_branch = bool(self.advanced_cfg.get("use_size_branch", False))
        self.use_dual_scale = bool(self.advanced_cfg.get("use_dual_scale", False))
        self.use_he_branch = bool(self.advanced_cfg.get("use_he_branch", False))

        size_in_dim = int(self.advanced_cfg.get("size_feature_dim", 5))
        self.size_branch = SizeMLP(in_dim=size_in_dim) if self.use_size_branch else None
        self.he_branch = HEBranch() if self.use_he_branch else None

        fusion_dim = self.image_encoder.num_features
        if self.use_dual_scale:
            fusion_dim += self.image_encoder.num_features
        if self.use_size_branch:
            fusion_dim += self.size_branch.out_dim
        if self.use_he_branch:
            fusion_dim += self.he_branch.out_dim

        self.classifier = nn.Linear(fusion_dim, num_classes)

        local_weight_path = cfg_model.get("local_weight_path", None)
        if local_weight_path:
            raw = torch.load(local_weight_path, map_location="cpu")
            state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("core."):
                    clean_state[k[len("core."):]] = v
                else:
                    clean_state[k] = v
            self.load_state_dict(clean_state, strict=False)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = FocalLoss(alpha=1.5, gamma=2.0, num_classes=num_classes)

        self.freeze_backbone_epochs = cfg_model.get("freeze_backbone_epochs", 0)
        self._frozen = False

    def maybe_freeze_backbone(self, current_epoch: int):
        if self.freeze_backbone_epochs <= 0:
            return
        if current_epoch < self.freeze_backbone_epochs and not self._frozen:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self._frozen = True
        if current_epoch >= self.freeze_backbone_epochs and self._frozen:
            for p in self.image_encoder.parameters():
                p.requires_grad = True
            self._frozen = False

    def _extract_inputs(self, batch):
        if isinstance(batch, torch.Tensor):
            return {"image": batch}
        if isinstance(batch, (list, tuple)):
            x, y = batch
            return {"image": x, "target": y}
        return batch

    def forward(self, batch):
        b = self._extract_inputs(batch)
        img_feat = self.image_encoder(self.mixstyle(b["image"]))
        feats = [img_feat]

        if self.use_dual_scale and "image_context" in b:
            ctx_feat = self.image_encoder(self.mixstyle(b["image_context"]))
            feats.append(ctx_feat)

        if self.use_size_branch and "size_features" in b:
            feats.append(self.size_branch(b["size_features"]))

        if self.use_he_branch and isinstance(b.get("he"), dict):
            he = torch.stack([b["he"]["H"], b["he"]["E"]], dim=1)
            feats.append(self.he_branch(he))

        fused = torch.cat(feats, dim=1)
        logits = self.classifier(fused)
        return logits

    def estimate_attention_map(self, batch):
        b = self._extract_inputs(batch)
        x = b["image"]
        attn = x.abs().mean(dim=1)
        attn = (attn - attn.amin(dim=(1, 2), keepdim=True)) / (attn.amax(dim=(1, 2), keepdim=True) - attn.amin(dim=(1, 2), keepdim=True) + 1e-6)
        return attn

    def border_attention_regularization(self, batch, lambda_border=0.0, lambda_center=0.0, border_ratio=0.15):
        attn = self.estimate_attention_map(batch)
        h, w = attn.shape[-2:]
        border, center = build_center_border_masks(h, w, border_ratio=border_ratio, device=attn.device)
        border_term = (attn * border).mean()
        center_term = (attn * center).mean()
        return lambda_border * border_term - lambda_center * center_term

    def _step(self, batch, stage: str):
        b = self._extract_inputs(batch)
        targets = b["target"]
        logits = self(batch)
        loss = self.criterion(logits, targets)
        return {"loss": loss, "logits": logits, "targets": targets}

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")
