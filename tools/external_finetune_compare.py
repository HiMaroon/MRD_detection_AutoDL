import argparse
import csv
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


import sys

def _resolve_project_root() -> Path:
    """Resolve repo root robustly across different launch directories."""
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if (c / "src").exists() and (c / "configs").exists():
            return c
    raise RuntimeError(
        f"无法定位项目根目录。请在仓库根目录运行，或确认存在 src/ 与 configs/。candidates={candidates}"
    )


project_root = _resolve_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.datasets import LabelFileDataset
from src.lit_module import LitSingleCell

def _load_yaml_cfg(name: str):
    cfg_path = project_root / "configs" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(cfg, resolve=True)


data_cfg = _load_yaml_cfg("data")
model_cfg = _load_yaml_cfg("model")

@dataclass
class EvalResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc: float
    num_samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_label_records(label_path: str) -> List[str]:
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and (not ln.strip().startswith("#"))]
    return lines


def save_label_records(lines: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def split_external(lines: List[str], tune_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    idx = list(range(len(lines)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_tune = max(1, int(len(lines) * tune_ratio))
    tune_idx = set(idx[:n_tune])
    tune_lines = [lines[i] for i in range(len(lines)) if i in tune_idx]
    holdout_lines = [lines[i] for i in range(len(lines)) if i not in tune_idx]
    if len(holdout_lines) == 0:
        raise ValueError("外部测试集过小，切分后 holdout 为空，请调小 --tune-ratio")
    return tune_lines, holdout_lines


def merge_morph_csv(csv_paths: List[Optional[str]], out_path: Path) -> Optional[str]:
    """Merge multiple morph CSV files into one file while deduplicating by image_path/filename.
    Later files override earlier files on key collision.
    """
    valid_paths = [p for p in csv_paths if p and os.path.exists(p)]
    if not valid_paths:
        return None

    merged: Dict[str, Dict[str, str]] = {}
    fieldnames = None

    for p in valid_paths:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                key = (row.get("image_path") or row.get("img_path") or row.get("path") or row.get("filename") or "").strip()
                if not key:
                    continue
                merged[key] = row

    if fieldnames is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged.values():
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return str(out_path)


def build_dataset(label_file: str, training: bool, morph_csv_path: str = None):
    return LabelFileDataset(
        label_file=label_file,
        img_size=data_cfg["img_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        augment=data_cfg.get("augment") if training else None,
        training=training,
        repeat_factor=(data_cfg.get("repeat_factor", 1) if training else 1),
        advanced_cfg=data_cfg.get("advanced", {}),
        return_morph=bool(data_cfg.get("return_morph", False)),
        morph_csv_path=morph_csv_path,
        morph_cols=data_cfg.get("morph_cols", ["area", "perimeter", "circularity"]),
    )


def make_loader(ds, batch_size: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> LitSingleCell:
    cfg = copy.deepcopy(model_cfg)
    cfg["local_weight_path"] = None
    model = LitSingleCell(cfg_model=cfg, num_classes=data_cfg["num_classes"], data_advanced_cfg=data_cfg.get("advanced", {}))

    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("core."):
            nk = k[len("core."):]
        remapped[nk] = v
    msg = model.load_state_dict(remapped, strict=False)
    print(f"[Load] {ckpt_path} | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.to(device)
    return model


def evaluate(model: LitSingleCell, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in ["image", "target", "morph", "morph_valid"]:
                if k in batch:
                    batch[k] = batch[k].to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            y_true.append(batch["target"].cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auc = np.nan
    try:
        num_classes = data_cfg["num_classes"]
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        pass

    return EvalResult(acc, p, r, f1, float(auc), int(len(y_true)))


def fine_tune(model: LitSingleCell, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float, weight_decay: float):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            for k in ["image", "target", "morph", "morph_valid"]:
                if k in batch:
                    batch[k] = batch[k].to(device)
            out = model.training_step(batch)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        print(f"[Epoch {ep + 1}/{epochs}] train_loss={np.mean(losses):.4f} val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1_macro:.4f}")
        if val_metrics.f1_macro > best_val:
            best_val = val_metrics.f1_macro
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def to_dict(res: EvalResult) -> Dict:
    return {
        "accuracy": res.accuracy,
        "precision_macro": res.precision_macro,
        "recall_macro": res.recall_macro,
        "f1_macro": res.f1_macro,
        "auc": res.auc,
        "num_samples": res.num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="在已有ckpt上混入少量外部测试集样本微调，并对剩余外部数据做前后指标对比")
    parser.add_argument("--ckpt", type=str, required=True, help="原始训练好的ckpt路径")
    parser.add_argument("--external-labels", type=str, nargs="+", required=True, help="外部测试集label txt路径（可一次给3个）")
    parser.add_argument("--external-names", type=str, nargs="+", default=None, help="外部测试集名称（与external-labels一一对应）")
    parser.add_argument("--base-train-labels", type=str, default=data_cfg["train_labels"], help="原始训练集label txt")
    parser.add_argument("--tune-ratio", type=float, default=0.1, help="从每个外部测试集抽取用于微调的比例")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--base-train-morph-csv", type=str, default=data_cfg.get("train_morph_csv"), help="原始训练集morph csv")
    parser.add_argument("--base-val-morph-csv", type=str, default=data_cfg.get("val_morph_csv"), help="评估时默认morph csv")
    parser.add_argument("--external-morph-csvs", type=str, nargs="+", default=None, help="外部测试集morph csv路径（与external-labels一一对应）")
    args = parser.parse_args()

    if args.external_names is not None and len(args.external_names) != len(args.external_labels):
        raise ValueError("--external-names 数量必须和 --external-labels 一致")
    if args.external_morph_csvs is not None and len(args.external_morph_csvs) != len(args.external_labels):
        raise ValueError("--external-morph-csvs 数量必须和 --external-labels 一致")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_train_lines = load_label_records(args.base_train_labels)

    all_summary = {}
    for i, ext_path in enumerate(args.external_labels):
        name = args.external_names[i] if args.external_names else Path(ext_path).stem
        exp_dir = outdir / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        ext_lines = load_label_records(ext_path)
        ext_morph_csv = args.external_morph_csvs[i] if args.external_morph_csvs else None
        tune_lines, holdout_lines = split_external(ext_lines, tune_ratio=args.tune_ratio, seed=args.seed + i)
        mixed_train_lines = base_train_lines + tune_lines

        mixed_train_path = exp_dir / "mixed_train_labels.txt"
        holdout_path = exp_dir / "holdout_labels.txt"
        save_label_records(mixed_train_lines, mixed_train_path)
        save_label_records(holdout_lines, holdout_path)

        holdout_morph_csv = merge_morph_csv([args.base_val_morph_csv, ext_morph_csv], exp_dir / "holdout_merged_morph.csv")
        holdout_ds = build_dataset(str(holdout_path), training=False, morph_csv_path=holdout_morph_csv)
        holdout_loader = make_loader(holdout_ds, batch_size=args.batch_size, shuffle=False)

        before_model = load_model_from_ckpt(args.ckpt, device)
        before_metrics = evaluate(before_model, holdout_loader, device)

        ft_model = load_model_from_ckpt(args.ckpt, device)
        train_morph_csv = merge_morph_csv([args.base_train_morph_csv, ext_morph_csv], exp_dir / "train_merged_morph.csv")
        ft_train_ds = build_dataset(str(mixed_train_path), training=True, morph_csv_path=train_morph_csv)
        ft_train_loader = make_loader(ft_train_ds, batch_size=args.batch_size, shuffle=True)

        ft_model = fine_tune(ft_model, ft_train_loader, holdout_loader, device, args.epochs, args.lr, args.weight_decay)
        after_metrics = evaluate(ft_model, holdout_loader, device)

        torch.save(ft_model.state_dict(), exp_dir / "finetuned_model_state_dict.pt")

        summary = {
            "dataset": name,
            "external_total": len(ext_lines),
            "tune_samples": len(tune_lines),
            "holdout_samples": len(holdout_lines),
            "before": to_dict(before_metrics),
            "after": to_dict(after_metrics),
            "delta": {
                "accuracy": after_metrics.accuracy - before_metrics.accuracy,
                "f1_macro": after_metrics.f1_macro - before_metrics.f1_macro,
                "auc": after_metrics.auc - before_metrics.auc if (not np.isnan(after_metrics.auc) and not np.isnan(before_metrics.auc)) else np.nan,
            },
        }
        all_summary[name] = summary

        with open(exp_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n===== {name} =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(outdir / "all_datasets_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

'''
python tools/external_finetune_compare.py \
  --ckpt /root/autodl-tmp/projects/myq/SingleCellProject/outputs/260501trial_2class_center_border_morph_loss_4features_lambda1/epoch=17-val_f1_macro=0.0000.ckpt \
  --external-labels \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH_labels_16.txt \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL_labels_16.txt \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU_labels_16.txt \
  --external-morph-csvs \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/morph_csv_norm/test_BJH_morph.csv \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/morph_csv_norm/test_FXH_noALL_morph.csv \
    /root/autodl-tmp/projects/myq/SingleCellProject/dataset/morph_csv_norm/test_TJMU_morph.csv \
  --external-names BJH FXH_noALL TJMU \
  --tune-ratio 0.1 \
  --epochs 3 \
  --batch-size 32 \
  --lr 1e-5 \
  --outdir /root/autodl-tmp/projects/myq/SingleCellProject/outputs/finetune_compare_outputs
'''