#无预训练 13版 lit_module.py 完整版（已加 label smoothing to Focal，smoothing=0.1）
import torch
import torch.nn as nn
import timm
# 在 lit_module.py 顶部加入（替换掉你之前手写的那个 FocalLoss）
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1.5, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha      # 正样本权重，稍后我们设 3~4
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         # logits: [B, 2], targets: [B] 长整型 0 或 1
#         # 加 label smoothing
#         targets_onehot = F.one_hot(targets, num_classes=2).float()
#         targets_smooth = targets_onehot * (1 - 0.15) + 0.15 / 2
#         ce_loss = F.cross_entropy(logits, targets_smooth, reduction='none')
#         pt = torch.exp(-ce_loss)                       # pt = p_t
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=2, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        # 统一处理alpha为tensor
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        elif isinstance(alpha, (int, float)):
            # 单个值 → 所有类别等权重
            self.alpha = torch.ones(num_classes, dtype=torch.float) * alpha
        else:
            raise TypeError(f"Unsupported alpha type: {type(alpha)}")

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 标签平滑
        if self.smoothing > 0:
            targets_smooth = targets_onehot * (1 - self.smoothing) + self.smoothing / self.num_classes
        else:
            targets_smooth = targets_onehot
        
        # 获取正确类别的概率 pt
        pt = (probs * targets_onehot).sum(dim=1).clamp(min=1e-7, max=1-1e-7)
        
        # 计算交叉熵
        ce_loss = -torch.sum(targets_smooth * torch.log(probs + 1e-7), dim=1)
        
        # 应用alpha权重
        if self.alpha is not None:
            # 确保alpha是tensor并移到正确设备
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0
        
        # Focal Loss
        focal_weight = (1 - pt) ** self.gamma
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score,
    binary_auroc,
)

class LitSingleCell(nn.Module):
    def __init__(self, cfg_model, num_classes, class_weights=None):
        super().__init__()

        self.cfg_model = cfg_model
        self.num_classes = num_classes

        # 创建模型
        self.model = timm.create_model(
            cfg_model["arch"],
            pretrained=cfg_model.get("pretrained", True),
            num_classes=num_classes,
            drop_rate=cfg_model.get("drop", 0.0),
            drop_path_rate=cfg_model.get("drop_path", 0.0),
        )

        # 加载本地预训练权重
        local_weight_path = cfg_model.get("local_weight_path", None)
        if local_weight_path:
            state_dict = torch.load(local_weight_path, map_location="cpu")
            state_dict.pop("classifier.weight", None)
            state_dict.pop("classifier.bias", None)
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)

            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"[Load Weights] {local_weight_path}")
            print("Missing keys:", len(msg.missing_keys))
            print("Unexpected keys:", len(msg.unexpected_keys))

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = FocalLoss(alpha=1.5, gamma=2.0, num_classes=num_classes)

        # Freeze backbone
        self.freeze_backbone_epochs = cfg_model.get("freeze_backbone_epochs", 0)
        self._frozen = False

    def maybe_freeze_backbone(self, current_epoch: int):
        if self.freeze_backbone_epochs <= 0:
            return

        if current_epoch < self.freeze_backbone_epochs and not self._frozen:
            for n, p in self.model.named_parameters():
                if "classifier" not in n and "fc" not in n and "head" not in n:
                    p.requires_grad = False
            self._frozen = True
            print(f"[Epoch {current_epoch}] Backbone frozen")

        if current_epoch >= self.freeze_backbone_epochs and self._frozen:
            for p in self.model.parameters():
                p.requires_grad = True
            self._frozen = False
            print(f"[Epoch {current_epoch}] Backbone unfrozen")

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        """
        返回: dict 包含 loss, logits, targets 用于上层计算多分类指标
        """
        x, y = batch  # y: [B], 类别索引 0,1,2,...,num_classes-1

        logits = self(x)  # [B, num_classes]
        loss = self.criterion(logits, y)

        x, y = batch
        logits = self.model(x)
        

        # 返回必要信息，让 Wrapper 计算多分类指标
        return {
            "loss": loss,
            "logits": logits,
            "targets": y,
        }

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")


# class LitSingleCell(nn.Module):
#     def __init__(self, cfg_model, num_classes, class_weights=None):
#         super().__init__()

#         self.cfg_model = cfg_model
#         self.num_classes = num_classes

#         # 创建模型（pretrained 可由 config 控制）
#         self.model = timm.create_model(
#             cfg_model["arch"],
#             pretrained=cfg_model.get("pretrained", True),
#             num_classes=num_classes,
#             drop_rate=cfg_model.get("drop", 0.0),
#             drop_path_rate=cfg_model.get("drop_path", 0.0),
#         )

#         # ====== 加载本地预训练权重（并打印加载情况）======
#         local_weight_path = cfg_model.get("local_weight_path", None)
#         if local_weight_path:
#             state_dict = torch.load(local_weight_path, map_location="cpu")

#             # 兼容旧 head（如果有的话）
#             state_dict.pop("classifier.weight", None)
#             state_dict.pop("classifier.bias", None)
#             state_dict.pop("fc.weight", None)
#             state_dict.pop("fc.bias", None)
#             state_dict.pop("head.weight", None)
#             state_dict.pop("head.bias", None)

#             msg = self.model.load_state_dict(state_dict, strict=False)
#             print(f"[Load Weights] {local_weight_path}")
#             print("Missing keys:", len(msg.missing_keys))
#             print("Unexpected keys:", len(msg.unexpected_keys))
#             # 想看具体 key 就取消下面两行注释
#             # print("Missing keys list:", msg.missing_keys)
#             # print("Unexpected keys list:", msg.unexpected_keys)

#         # loss 采用Focalloss
#         self.criterion = FocalLoss(alpha=1.5, gamma=2.0)
#         #改回原来的loss
#         #self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to('cuda'))

#         # freeze 相关
#         self.freeze_backbone_epochs = cfg_model.get("freeze_backbone_epochs", 0)
#         self._frozen = False

#     def maybe_freeze_backbone(self, current_epoch: int):
#         if self.freeze_backbone_epochs <= 0:
#             return

#         if current_epoch < self.freeze_backbone_epochs and not self._frozen:
#             for n, p in self.model.named_parameters():
#                 if "classifier" not in n and "fc" not in n and "head" not in n:
#                     p.requires_grad = False
#             self._frozen = True

#         if current_epoch >= self.freeze_backbone_epochs and self._frozen:
#             for p in self.model.parameters():
#                 p.requires_grad = True
#             self._frozen = False

#     def forward(self, x):
#         return self.model(x)

#     def _step(self, batch, stage: str):
#         x, y = batch  # y: [B], 0/1

#         logits = self(x)
#         loss = self.criterion(logits, y)

#         probs = torch.softmax(logits, dim=1)[:, 1]
#         preds = (probs >= 0.4).long()

#         acc = binary_accuracy(preds, y)
#         prec = binary_precision(preds, y)
#         rec = binary_recall(preds, y)
#         f1 = binary_f1_score(preds, y)
#         auroc = binary_auroc(probs, y)

#         metrics = {
#             f"{stage}/loss": loss.detach(),
#             f"{stage}/acc": acc.detach(),
#             f"{stage}/prec": prec.detach(),
#             f"{stage}/rec": rec.detach(),
#             f"{stage}/f1": f1.detach(),
#             f"{stage}/auroc": auroc.detach(),
#         }
#         return loss, metrics

#     def training_step(self, batch):
#         return self._step(batch, "train")

#     def validation_step(self, batch):
#         return self._step(batch, "val")
