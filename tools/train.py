import os
import sys
import csv
import math
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_confusion_matrix,
)
from torchmetrics.functional import auroc


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(_cfg):
    original_cwd = hydra.utils.get_original_cwd()
    print(f"Hydra original config path: {original_cwd}")

    if original_cwd not in sys.path:
        sys.path.append(original_cwd)

    from configs import data_cfg, model_cfg, train_cfg, wandb_cfg
    from src.datamodule import SingleCellDataModule
    from src.lit_module import LitSingleCell
    from src.utils import set_seed

    data_cfg_ = data_cfg
    model_cfg_ = model_cfg
    train_cfg_ = train_cfg
    wandb_cfg_ = wandb_cfg

    out_root = train_cfg_["output_root"]
    os.makedirs(out_root, exist_ok=True)
    set_seed(train_cfg_["seed"])

    dm = SingleCellDataModule(
        data_cfg_,
        batch_size=train_cfg_["batch_size"],
        num_workers=data_cfg_["num_workers"],
        pin_memory=data_cfg_["pin_memory"],
    )
    dm.setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = data_cfg_["num_classes"]

    core = LitSingleCell(
        model_cfg_,
        num_classes=num_classes,
    )

    # ====== Freeze backbone 回调 ======
    class FreezeCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if hasattr(pl_module, "core") and hasattr(pl_module.core, "maybe_freeze_backbone"):
                pl_module.core.maybe_freeze_backbone(trainer.current_epoch)

    # ---------- Lightning Wrapper ----------
    class _Wrapper(pl.LightningModule):
        def __init__(self, core, num_classes, train_cfg, out_root):
            super().__init__()
            self.core = core
            self.num_classes = num_classes
            self.train_cfg = train_cfg
            self.out_root = out_root

            # 收集每个 batch 输出，用于 epoch 末汇总
            self.train_logits = []
            self.train_targets = []
            self.val_logits = []
            self.val_targets = []

            # 当前 epoch 的 train/val 汇总缓存
            self.epoch_metric_buffer = {}

            # 本地 metrics.csv 路径
            self.metrics_csv_path = os.path.join(self.out_root, "metrics.csv")

        def forward(self, x):
            return self.core(x)

        def _to_python(self, x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                if x.numel() == 1:
                    v = x.item()
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    return v
                return x.tolist()
            return x

        def _safe_binary_auroc(self, probs_1d, binary_targets):
            """
            对单个 one-vs-rest 任务计算 binary AUROC
            若当前 epoch 内正样本或负样本缺失，则返回 None
            """
            pos_count = binary_targets.sum()
            total = len(binary_targets)

            if pos_count == 0 or pos_count == total:
                return None

            try:
                from torchmetrics.functional.classification import binary_auroc
                return binary_auroc(probs_1d, binary_targets)
            except (ImportError, TypeError):
                return auroc(probs_1d, binary_targets, task="binary")

        def _compute_auroc(self, probs, targets, average="macro"):
            """
            统一 AUROC 计算
            average:
                - macro: 各类别 AUROC 简单平均
                - weighted: 按各类别 support 加权平均
            """
            if self.num_classes == 2:
                try:
                    from torchmetrics.functional.classification import binary_auroc
                    return binary_auroc(probs[:, 1], targets)
                except (ImportError, TypeError):
                    return auroc(probs[:, 1], targets, task="binary")

            aurocs = []
            supports = []

            for i in range(self.num_classes):
                binary_targets = (targets == i).long()
                auroc_i = self._safe_binary_auroc(probs[:, i], binary_targets)
                if auroc_i is None:
                    continue

                aurocs.append(auroc_i)
                supports.append(binary_targets.sum().float())

            if len(aurocs) == 0:
                return torch.tensor(float("nan"), device=probs.device)

            aurocs = torch.stack(aurocs)
            supports = torch.stack(supports)

            if average == "weighted":
                weights = supports / supports.sum()
                return (aurocs * weights).sum()

            return aurocs.mean()

        def _compute_classification_metrics(self, preds, targets, prefix="", sync_dist=False):
            """
            计算并记录 macro / weighted / per-class 分类指标
            """
            # Macro
            acc_macro = multiclass_accuracy(
                preds, targets, self.num_classes, average="macro"
            )
            prec_macro = multiclass_precision(
                preds, targets, self.num_classes, average="macro"
            )
            rec_macro = multiclass_recall(
                preds, targets, self.num_classes, average="macro"
            )
            f1_macro = multiclass_f1_score(
                preds, targets, self.num_classes, average="macro"
            )

            # Weighted
            acc_weighted = multiclass_accuracy(
                preds, targets, self.num_classes, average="weighted"
            )
            prec_weighted = multiclass_precision(
                preds, targets, self.num_classes, average="weighted"
            )
            rec_weighted = multiclass_recall(
                preds, targets, self.num_classes, average="weighted"
            )
            f1_weighted = multiclass_f1_score(
                preds, targets, self.num_classes, average="weighted"
            )

            # Per-class
            acc_per_class = multiclass_accuracy(
                preds, targets, self.num_classes, average=None
            )
            prec_per_class = multiclass_precision(
                preds, targets, self.num_classes, average=None
            )
            rec_per_class = multiclass_recall(
                preds, targets, self.num_classes, average=None
            )
            f1_per_class = multiclass_f1_score(
                preds, targets, self.num_classes, average=None
            )

            # Lightning logger 记录为 epoch 级别
            self.log(f"{prefix}/acc_macro", acc_macro, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/prec_macro", prec_macro, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/rec_macro", rec_macro, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/f1_macro", f1_macro, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)

            self.log(f"{prefix}/acc_weighted", acc_weighted, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/prec_weighted", prec_weighted, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/rec_weighted", rec_weighted, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/f1_weighted", f1_weighted, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)

            for i in range(self.num_classes):
                self.log(f"{prefix}/acc_class_{i}", acc_per_class[i], on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
                self.log(f"{prefix}/prec_class_{i}", prec_per_class[i], on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
                self.log(f"{prefix}/rec_class_{i}", rec_per_class[i], on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
                self.log(f"{prefix}/f1_class_{i}", f1_per_class[i], on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)

            return {
                "acc_macro": acc_macro,
                "prec_macro": prec_macro,
                "rec_macro": rec_macro,
                "f1_macro": f1_macro,
                "acc_weighted": acc_weighted,
                "prec_weighted": prec_weighted,
                "rec_weighted": rec_weighted,
                "f1_weighted": f1_weighted,
                "acc_per_class": acc_per_class,
                "prec_per_class": prec_per_class,
                "rec_per_class": rec_per_class,
                "f1_per_class": f1_per_class,
            }

        def _append_metrics_to_csv(self, row_dict):
            """
            每个 epoch 写一行到 metrics.csv
            自动展开 list 字段为 xxx_0, xxx_1, ...
            """
            flat_row = {}
            for k, v in row_dict.items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        flat_row[f"{k}_{i}"] = item
                else:
                    flat_row[k] = v

            file_exists = os.path.exists(self.metrics_csv_path)

            if file_exists:
                with open(self.metrics_csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    try:
                        existing_header = next(reader)
                    except StopIteration:
                        existing_header = []
                fieldnames = list(existing_header)
                for k in flat_row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            else:
                fieldnames = list(flat_row.keys())

            old_rows = []
            if file_exists:
                with open(self.metrics_csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    old_rows = list(reader)

            for old_row in old_rows:
                for fn in fieldnames:
                    if fn not in old_row:
                        old_row[fn] = ""

            for fn in fieldnames:
                if fn not in flat_row:
                    flat_row[fn] = ""

            with open(self.metrics_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in old_rows:
                    writer.writerow(r)
                writer.writerow(flat_row)

        def _flush_epoch_metrics(self):
            """
            在一个完整 epoch 的 train/val 都结束后：
            1. 保存到本地 metrics.csv
            2. 上传到 wandb，并强制按 epoch 记录
            """
            if not self.epoch_metric_buffer:
                return

            row = {
                "epoch": int(self.current_epoch),
                **self.epoch_metric_buffer,
            }

            # 本地保存
            self._append_metrics_to_csv(row)

            # WandB 按 epoch 保存
            if self.train_cfg.get("use_wandb", False):
                try:
                    import wandb

                    wandb_row = {}
                    for k, v in row.items():
                        if isinstance(v, list):
                            for i, item in enumerate(v):
                                wandb_row[f"{k}_{i}"] = item
                        else:
                            wandb_row[k] = v

                    # 关键：epoch 作为显式横轴
                    wandb_row["epoch"] = int(self.current_epoch)

                    # 用 epoch 作为 step
                    wandb.log(wandb_row, step=int(self.current_epoch))
                except Exception as e:
                    print(f"⚠️ WandB epoch 指标写入失败：{e}")

        def _finalize_epoch_metrics(self, logits_list, targets_list, prefix="train", sync_dist=False):
            """
            在 epoch 结束时统一汇总并计算指标
            """
            if len(logits_list) == 0:
                return

            all_logits = torch.cat(logits_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            all_preds = torch.argmax(all_logits, dim=1)
            all_probs = torch.softmax(all_logits, dim=1)

            metric_dict = self._compute_classification_metrics(
                all_preds, all_targets, prefix=prefix, sync_dist=sync_dist
            )

            auroc_macro = self._compute_auroc(all_probs, all_targets, average="macro")
            auroc_weighted = self._compute_auroc(all_probs, all_targets, average="weighted")

            self.log(f"{prefix}/auroc_macro", auroc_macro, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)
            self.log(f"{prefix}/auroc_weighted", auroc_weighted, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist)

            if prefix == "val" and (self.current_epoch % 5 == 0 or self.current_epoch == 0):
                cm = multiclass_confusion_matrix(all_preds, all_targets, self.num_classes)
                print(f"\n[Epoch {self.current_epoch}] Confusion Matrix:\n{cm}")

            # 存入当前 epoch buffer，稍后统一写 csv / wandb
            record = {
                f"{prefix}/acc_macro": self._to_python(metric_dict["acc_macro"]),
                f"{prefix}/prec_macro": self._to_python(metric_dict["prec_macro"]),
                f"{prefix}/rec_macro": self._to_python(metric_dict["rec_macro"]),
                f"{prefix}/f1_macro": self._to_python(metric_dict["f1_macro"]),
                f"{prefix}/acc_weighted": self._to_python(metric_dict["acc_weighted"]),
                f"{prefix}/prec_weighted": self._to_python(metric_dict["prec_weighted"]),
                f"{prefix}/rec_weighted": self._to_python(metric_dict["rec_weighted"]),
                f"{prefix}/f1_weighted": self._to_python(metric_dict["f1_weighted"]),
                f"{prefix}/auroc_macro": self._to_python(auroc_macro),
                f"{prefix}/auroc_weighted": self._to_python(auroc_weighted),
                f"{prefix}/acc_per_class": self._to_python(metric_dict["acc_per_class"]),
                f"{prefix}/prec_per_class": self._to_python(metric_dict["prec_per_class"]),
                f"{prefix}/rec_per_class": self._to_python(metric_dict["rec_per_class"]),
                f"{prefix}/f1_per_class": self._to_python(metric_dict["f1_per_class"]),
            }
            self.epoch_metric_buffer.update(record)

            print(
                f"[Epoch {self.current_epoch}] {prefix.capitalize()} | "
                f"Acc(macro): {metric_dict['acc_macro']:.4f} | "
                f"Acc(weighted): {metric_dict['acc_weighted']:.4f} | "
                f"Prec(macro): {metric_dict['prec_macro']:.4f} | "
                f"Prec(weighted): {metric_dict['prec_weighted']:.4f} | "
                f"Rec(macro): {metric_dict['rec_macro']:.4f} | "
                f"Rec(weighted): {metric_dict['rec_weighted']:.4f} | "
                f"F1(macro): {metric_dict['f1_macro']:.4f} | "
                f"F1(weighted): {metric_dict['f1_weighted']:.4f} | "
                f"AUROC(macro): {auroc_macro:.4f} | "
                f"AUROC(weighted): {auroc_weighted:.4f}"
            )
            print(
                f"Per-class Acc: {[f'{x:.3f}' for x in metric_dict['acc_per_class'].tolist()]}"
            )

        def training_step(self, batch, batch_idx):
            out = self.core.training_step(batch)
            loss = out["loss"]
            logits = out["logits"]
            targets = out["targets"]

            self.train_logits.append(logits.detach().cpu())
            self.train_targets.append(targets.detach().cpu())

            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            out = self.core.validation_step(batch)
            loss = out["loss"]
            logits = out["logits"]
            targets = out["targets"]

            self.val_logits.append(logits.detach().cpu())
            self.val_targets.append(targets.detach().cpu())

            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
            return loss

        def on_train_epoch_end(self):
            self._finalize_epoch_metrics(
                self.train_logits,
                self.train_targets,
                prefix="train",
                sync_dist=False,
            )
            self.train_logits.clear()
            self.train_targets.clear()

        def on_validation_epoch_end(self):
            self._finalize_epoch_metrics(
                self.val_logits,
                self.val_targets,
                prefix="val",
                sync_dist=True,
            )
            self.val_logits.clear()
            self.val_targets.clear()

            # 一个完整 epoch 的 train/val 指标齐了，这里统一写
            self._flush_epoch_metrics()

            # 清空缓存，准备下一个 epoch
            self.epoch_metric_buffer.clear()

        def configure_optimizers(self):
            import torch.optim as optim

            params = self.core.parameters()
            if self.train_cfg["optimizer"]["name"] == "sgd":
                opt = optim.SGD(
                    params,
                    lr=self.train_cfg["optimizer"]["lr"],
                    momentum=self.train_cfg["optimizer"].get("momentum", 0.9),
                    weight_decay=self.train_cfg["optimizer"]["weight_decay"],
                )
            else:
                opt = optim.AdamW(
                    params,
                    lr=self.train_cfg["optimizer"]["lr"],
                    weight_decay=self.train_cfg["optimizer"]["weight_decay"],
                )

            if self.train_cfg["scheduler"]["name"] == "cosine":
                T_max = max(
                    1,
                    self.train_cfg["max_epochs"] - self.train_cfg["scheduler"]["warmup_epochs"],
                )
                sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
                return {"optimizer": opt, "lr_scheduler": sch}

            return opt

    # ---------- Callbacks ----------
    cbs = [
        FreezeCallback(),
        EarlyStopping(**train_cfg_["callbacks"]["early_stopping"]),
        ModelCheckpoint(dirpath=out_root, **train_cfg_["callbacks"]["checkpoint"]),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ---------- Logger ----------
    if train_cfg_.get("use_wandb", False):
        import wandb
        import yaml
        from datetime import datetime

        logger = WandbLogger(
            project=wandb_cfg_["project"],
            entity=wandb_cfg_.get("entity", None),
            mode=wandb_cfg_.get("mode", "disabled"),
            name=wandb_cfg_.get("name", f"run_{os.path.basename(out_root)}"),
            tags=wandb_cfg_.get("tags", []),
            save_dir=out_root,
            log_model=False,
        )

        if logger.experiment is not None:
            # 让 wandb 知道 epoch 是横轴
            try:
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("val/*", step_metric="epoch")
            except Exception as e:
                print(f"⚠️ wandb.define_metric 设置失败：{e}")

            full_config = {
                "data": data_cfg_,
                "model": model_cfg_,
                "train": train_cfg_,
                "wandb": wandb_cfg_,
                "runtime": {
                    "num_classes": num_classes,
                    "output_root": out_root,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "cuda_available": torch.cuda.is_available(),
                    "device": device,
                },
            }

            logger.log_hyperparams(full_config)

            config_paths = []
            for cfg_name, cfg_obj in [
                ("config_data", data_cfg_),
                ("config_model", model_cfg_),
                ("config_train", train_cfg_),
                ("config_wandb", wandb_cfg_),
            ]:
                cfg_path = os.path.join(out_root, f"{cfg_name}.yaml")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg_obj, f, default_flow_style=False, allow_unicode=True)
                config_paths.append(cfg_path)

            try:
                artifact = wandb.Artifact(
                    name=f"config_{wandb_cfg_.get('name', os.path.basename(out_root))}",
                    type="config",
                    description="Complete training configuration",
                )
                for cfg_path in config_paths:
                    artifact.add_file(cfg_path)
                logger.experiment.log_artifact(artifact)
            except Exception as e:
                print(f"⚠️ Artifact 上传失败：{e}")

            print(f"\n✅ WandB 配置已保存:")
            print(f"   Run URL: {logger.experiment.url}")
            print(f"   Config files: {out_root}/*.yaml")
            print(f"   Metrics CSV: {os.path.join(out_root, 'metrics.csv')}\n")
    else:
        logger = True
        print(f"\n✅ 本地 Metrics CSV 将保存到: {os.path.join(out_root, 'metrics.csv')}\n")

    trainer = pl.Trainer(
        max_epochs=train_cfg_["max_epochs"],
        accelerator="auto",
        devices=train_cfg_["devices"],
        precision=train_cfg_["precision"],
        accumulate_grad_batches=train_cfg_["accumulate_grad_batches"],
        callbacks=cbs,
        logger=logger,
        log_every_n_steps=train_cfg_.get("log_every_n_steps", 50),
        num_sanity_val_steps=0,
    )

    model = _Wrapper(core, num_classes, train_cfg_, out_root)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()