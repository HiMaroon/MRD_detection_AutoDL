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


class LitSingleCell(nn.Module):
    def __init__(self, cfg_model, num_classes, class_weights=None, data_advanced_cfg=None):
        super().__init__()
        self.cfg_model = cfg_model
        self.num_classes = num_classes
        self.advanced_cfg = data_advanced_cfg or {}

        # Always define these early so later callbacks can safely access them.
        self.freeze_backbone_epochs = int(cfg_model.get("freeze_backbone_epochs", 0) or 0)
        self._frozen = False

        arch = cfg_model["arch"]
        pretrained = cfg_model.get("pretrained", True)
        drop = cfg_model.get("drop", 0.0)
        drop_path = cfg_model.get("drop_path", 0.0)

        self.image_encoder = ImageEncoder(arch, pretrained, drop, drop_path)
        self.mixstyle = MixStyle(
            p=cfg_model.get("mixstyle_prob", 0.0),
            alpha=cfg_model.get("mixstyle_alpha", 0.1),
        )

        fusion_dim = self.image_encoder.num_features
        self.morph_dim = int(cfg_model.get("morph_dim", 3))
        self.morph_hidden = int(cfg_model.get("morph_hidden", 128))
        self.lambda_morph = float(cfg_model.get("lambda_morph", 0.05))
        self.morph_loss_name = str(cfg_model.get("morph_loss", "smoothl1")).lower()
        if self.morph_loss_name not in {"smoothl1", "mse"}:
            raise ValueError(f"Unsupported morph_loss={self.morph_loss_name}, expected one of ['smoothl1', 'mse']")

        # morph_mode: none / loss / fusion / loss+fusion
        self.morph_mode = str(cfg_model.get("morph_mode", "loss" if cfg_model.get("use_morph_head", False) else "none")).lower()
        valid_modes = {"none", "loss", "fusion", "loss+fusion"}
        if self.morph_mode not in valid_modes:
            raise ValueError(f"Unsupported morph_mode={self.morph_mode}, expected one of {sorted(valid_modes)}")
        self.use_morph_head = self.morph_mode in {"loss", "loss+fusion"}
        self.use_morph_fusion = self.morph_mode in {"fusion", "loss+fusion"}
        self.morph_fusion_hidden = int(cfg_model.get("morph_fusion_hidden", self.morph_hidden))

        if self.use_morph_fusion:
            self.morph_fusion = nn.Sequential(
                nn.Linear(self.morph_dim, self.morph_fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            )
            classifier_in_dim = fusion_dim + self.morph_fusion_hidden
        else:
            self.morph_fusion = None
            classifier_in_dim = fusion_dim

        # New task-specific classifier head.
        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        if self.use_morph_head:
            self.morph_head = nn.Sequential(
                nn.Linear(fusion_dim, self.morph_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
                nn.Linear(self.morph_hidden, self.morph_dim),
            )
        else:
            self.morph_head = None

        local_weight_path = cfg_model.get("local_weight_path", None)
        if local_weight_path:
            raw = torch.load(local_weight_path, map_location="cpu")
            state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("core.image_encoder.backbone."):
                    nk = k[len("core.image_encoder.backbone."):]
                elif k.startswith("image_encoder.backbone."):
                    nk = k[len("image_encoder.backbone."):]
                elif k.startswith("core.model."):
                    nk = k[len("core.model."):]
                elif k.startswith("model."):
                    nk = k[len("model."):]
                else:
                    nk = k
                clean_state[nk] = v

            for head_k in [
                "classifier.weight", "classifier.bias",
                "fc.weight", "fc.bias",
                "head.weight", "head.bias",
                "head.fc.weight", "head.fc.bias",
            ]:
                clean_state.pop(head_k, None)

            msg = self.image_encoder.backbone.load_state_dict(clean_state, strict=False)

            print(f"[Load Weights] {local_weight_path}")
            print(f"  missing_keys={len(msg.missing_keys)} | unexpected_keys={len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print("  missing keys (first 20):")
                for k in msg.missing_keys[:20]:
                    print(f"    - {k}")
            if msg.unexpected_keys:
                print("  unexpected keys (first 20):")
                for k in msg.unexpected_keys[:20]:
                    print(f"    - {k}")

        # Loss must be defined regardless of whether local weights are used.
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = FocalLoss(alpha=1.5, gamma=2.0, num_classes=num_classes)

    def maybe_freeze_backbone(self, current_epoch: int):
        freeze_backbone_epochs = getattr(self, "freeze_backbone_epochs", 0)
        if freeze_backbone_epochs <= 0:
            return

        if current_epoch < freeze_backbone_epochs and not self._frozen:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self._frozen = True
            print(f"[Epoch {current_epoch}] Backbone frozen")

        if current_epoch >= freeze_backbone_epochs and self._frozen:
            for p in self.image_encoder.parameters():
                p.requires_grad = True
            self._frozen = False
            print(f"[Epoch {current_epoch}] Backbone unfrozen")

    def _extract_inputs(self, batch):
        if isinstance(batch, torch.Tensor):
            return {"image": batch}
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                return {"image": x, "target": y}
            if len(batch) == 4:
                x, y, morph, morph_valid = batch
                return {"image": x, "target": y, "morph": morph, "morph_valid": morph_valid}
            raise ValueError(f"Unexpected tuple/list batch format: len(batch)={len(batch)}")
        return batch

    def extract_feat(self, x):
        return self.image_encoder(self.mixstyle(x))

    def _build_classifier_input(self, img_feat, morph=None, morph_valid=None):
        if not self.use_morph_fusion:
            return img_feat

        if morph is None:
            morph = torch.zeros(img_feat.size(0), self.morph_dim, device=img_feat.device, dtype=img_feat.dtype)
        morph = morph.to(img_feat.device).float()
        if morph_valid is not None:
            morph_valid = morph_valid.to(img_feat.device).float()
            morph = morph * morph_valid
        morph_feat = self.morph_fusion(morph)
        return torch.cat([img_feat, morph_feat], dim=1)

    def forward(self, batch):
        b = self._extract_inputs(batch)
        img_feat = self.extract_feat(b["image"])
        cls_in = self._build_classifier_input(img_feat, b.get("morph"), b.get("morph_valid"))
        logits = self.classifier(cls_in)
        return logits

    def forward_all(self, batch):
        b = self._extract_inputs(batch)
        img_feat = self.extract_feat(b["image"])
        cls_in = self._build_classifier_input(img_feat, b.get("morph"), b.get("morph_valid"))
        logits = self.classifier(cls_in)
        out = {"feat": img_feat, "logits": logits}
        if self.use_morph_head:
            out["morph_pred"] = self.morph_head(img_feat)
        return out

    def estimate_attention_map(self, batch):
        b = self._extract_inputs(batch)
        x = b["image"]
        attn = x.abs().mean(dim=1)
        attn = (attn - attn.amin(dim=(1, 2), keepdim=True)) / (
            attn.amax(dim=(1, 2), keepdim=True) - attn.amin(dim=(1, 2), keepdim=True) + 1e-6
        )
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
        all_out = self.forward_all(batch)
        logits = all_out["logits"]
        loss = self.criterion(logits, targets)

        morph_loss = torch.tensor(0.0, device=logits.device)
        morph_coverage = None
        has_morph_target = ("morph" in b) and ("morph_valid" in b)
        if has_morph_target:
            morph_valid = b["morph_valid"].to(logits.device).float()
            morph_coverage = morph_valid.mean()

        if self.use_morph_head and has_morph_target and ("morph_pred" in all_out):
            morph = b["morph"].to(logits.device).float()
            morph_pred = all_out["morph_pred"]

            if self.morph_loss_name == "mse":
                per_elem_loss = F.mse_loss(morph_pred, morph, reduction="none")
            else:
                per_elem_loss = F.smooth_l1_loss(morph_pred, morph, reduction="none")

            denom = morph_valid.sum().clamp(min=1.0)
            morph_loss = (per_elem_loss * morph_valid).sum() / denom
            loss = loss + self.lambda_morph * morph_loss

        return {"loss": loss, "logits": logits, "targets": targets, "morph_loss": morph_loss, "morph_coverage": morph_coverage}

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")
