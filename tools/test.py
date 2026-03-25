# tools/test.py
import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# 项目根目录调整
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datamodule import SingleCellDataModule
from src.lit_module import LitSingleCell
from configs import data_cfg, model_cfg, train_cfg


class _Wrapper(pl.LightningModule):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return self.core(x)


def format_confusion_matrix(cm, class_labels):
    """
    将混淆矩阵格式化为可读文本，适用于任意类别数
    """
    header = ["True\\Pred"] + [str(c) for c in class_labels]
    col_width = max(10, max(len(h) for h in header) + 2)

    lines = []
    lines.append("".join(h.rjust(col_width) for h in header))
    for i, row in enumerate(cm):
        row_text = [str(class_labels[i])] + [str(v) for v in row]
        lines.append("".join(x.rjust(col_width) for x in row_text))
    return lines


def calculate_and_save_metrics(df, output_dir, split, timestamp, num_classes):
    """
    计算分类指标并保存到 TXT 文件，兼容二分类 / 多分类
    """
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    prob_cols = [f"prob_class_{i}" for i in range(num_classes) if f"prob_class_{i}" in df.columns]
    y_prob = df[prob_cols].values if len(prob_cols) > 0 else None

    metrics = {}
    report_text = []

    class_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # 1. 基础指标
    acc = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = float(acc)

    report_text.append("=" * 80)
    report_text.append(f"📊 {split.upper()} SET EVALUATION METRICS")
    report_text.append("=" * 80)
    report_text.append(f"Total Samples: {len(df)}")
    report_text.append(f"Num Classes (configured): {num_classes}")
    report_text.append(f"Observed Classes: {class_labels}")
    report_text.append(f"Class Distribution (True): {dict(zip(*np.unique(y_true, return_counts=True)))}")
    report_text.append(f"Class Distribution (Pred): {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    report_text.append(f"Accuracy: {acc:.4f}")
    report_text.append("-" * 80)

    # 2. classification_report
    try:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            output_dict=True,
            zero_division=0
        )

        report_text.append("\n📋 PER-CLASS METRICS:")
        report_text.append("-" * 80)

        for cls in range(num_classes):
            cls_key = str(cls)
            if cls_key not in report_dict:
                continue

            cls_metrics = report_dict[cls_key]
            prec = cls_metrics["precision"]
            rec = cls_metrics["recall"]
            f1 = cls_metrics["f1-score"]
            support = cls_metrics["support"]

            metrics[f"class_{cls}_precision"] = float(prec)
            metrics[f"class_{cls}_recall"] = float(rec)
            metrics[f"class_{cls}_f1"] = float(f1)
            metrics[f"class_{cls}_support"] = int(support)

            report_text.append(f"Class {cls}:")
            report_text.append(f"  Precision:  {prec:.4f}")
            report_text.append(f"  Recall:     {rec:.4f}")
            report_text.append(f"  F1-Score:   {f1:.4f}")
            report_text.append(f"  Support:    {support}")
            report_text.append("")

        report_text.append("📋 OVERALL METRICS:")
        report_text.append("-" * 80)

        for avg_name in ["macro avg", "weighted avg"]:
            if avg_name in report_dict:
                avg_prefix = avg_name.replace(" ", "_")
                avg_prec = report_dict[avg_name]["precision"]
                avg_rec = report_dict[avg_name]["recall"]
                avg_f1 = report_dict[avg_name]["f1-score"]

                metrics[f"{avg_prefix}_precision"] = float(avg_prec)
                metrics[f"{avg_prefix}_recall"] = float(avg_rec)
                metrics[f"{avg_prefix}_f1"] = float(avg_f1)

                report_text.append(
                    f"{avg_name} - Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}"
                )

    except Exception as e:
        report_text.append(f"⚠️ Classification Report 计算警告：{e}")

    # 3. AUC
    report_text.append("")
    report_text.append("📋 AUC METRICS:")
    report_text.append("-" * 80)

    try:
        if y_prob is None or len(prob_cols) != num_classes:
            report_text.append("AUC: N/A (概率列不完整)")
            metrics["auc_roc"] = None
        else:
            unique_true = np.unique(y_true)

            if len(unique_true) < 2:
                report_text.append("AUC: N/A (y_true 只有一个类别)")
                metrics["auc_roc"] = None

            elif num_classes == 2:
                # 二分类：取正类概率
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
                metrics["auc_roc"] = float(auc_roc)
                report_text.append(f"ROC-AUC (binary): {auc_roc:.4f}")

            else:
                # 多分类：One-vs-Rest
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

                auc_macro_ovr = roc_auc_score(
                    y_true_bin,
                    y_prob,
                    average="macro",
                    multi_class="ovr"
                )
                auc_weighted_ovr = roc_auc_score(
                    y_true_bin,
                    y_prob,
                    average="weighted",
                    multi_class="ovr"
                )

                metrics["auc_roc_macro_ovr"] = float(auc_macro_ovr)
                metrics["auc_roc_weighted_ovr"] = float(auc_weighted_ovr)

                report_text.append(f"ROC-AUC Macro OVR:    {auc_macro_ovr:.4f}")
                report_text.append(f"ROC-AUC Weighted OVR: {auc_weighted_ovr:.4f}")

    except Exception as e:
        report_text.append(f"AUC: 计算错误 ({e})")
        metrics["auc_roc"] = None

    # 4. 混淆矩阵
    report_text.append("")
    report_text.append("📋 CONFUSION MATRIX:")
    report_text.append("-" * 80)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    cm_lines = format_confusion_matrix(cm, class_labels=list(range(num_classes)))
    report_text.extend(cm_lines)

    # 二分类额外输出 TN/FP/FN/TP
    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negative"] = int(tn)
        metrics["false_positive"] = int(fp)
        metrics["false_negative"] = int(fn)
        metrics["true_positive"] = int(tp)

        report_text.append("")
        report_text.append("📋 CONFUSION MATRIX DETAILS:")
        report_text.append(f"  TN (真阴性): {tn}")
        report_text.append(f"  FP (假阳性): {fp}")
        report_text.append(f"  FN (假阴性): {fn}")
        report_text.append(f"  TP (真阳性): {tp}")

    report_text.append("=" * 80)

    # 5. 保存
    final_out_dir = Path(output_dir)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = final_out_dir / f"{split}_metrics_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    # 可选：再保存一份 json，方便后处理
    json_path = final_out_dir / f"{split}_metrics_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return txt_path, json_path


def run_test_on_split(split: str = "val", ckpt_path: str = None, test_data_sir: str = None, output_dir: str = None):
    assert split in ["train", "val"]

    # === 1. 验证 Checkpoint ===
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到指定的 checkpoint: {ckpt_path}")
    print(f"🚀 使用 checkpoint: {ckpt_path}")

    # === 2. 加载模型 ===
    num_classes = data_cfg["num_classes"]
    core = LitSingleCell(model_cfg, num_classes=num_classes)
    model = _Wrapper.load_from_checkpoint(ckpt_path, core=core, map_location="cpu")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === 3. 配置 DataModule 路径 ===
    test_data_cfg = data_cfg.copy()
    test_data_cfg["val_labels"] = test_data_sir

    dm = SingleCellDataModule(
        test_data_cfg,
        batch_size=train_cfg.get("batch_size", 64),
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    dm.setup(stage=None)

    loader = dm.train_dataloader() if split == "train" else dm.val_dataloader()
    dataset = loader.dataset
    print(f"📊 [{split.upper()} SET] 样本总数：{len(dataset)} | BatchSize: {loader.batch_size} | NumClasses: {num_classes}")

    # === 4. 推理核心逻辑 ===
    all_results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), loader.batch_size), desc=f"Testing {split}", ncols=100):
            batch_indices = range(i, min(i + loader.batch_size, len(dataset)))

            batch_imgs = []
            batch_targets = []
            batch_paths = []

            for idx in batch_indices:
                img, label = dataset[idx]
                path, _ = dataset.samples[idx]
                batch_imgs.append(img)
                batch_targets.append(label)
                batch_paths.append(path)

            x = torch.stack(batch_imgs).to(device)
            y = torch.tensor(batch_targets).to(device)

            logits = model.core(x)
            probs = torch.softmax(logits, dim=1)   # [B, C]
            preds = torch.argmax(probs, dim=1)     # [B]

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = y.cpu().numpy()

            for j in range(len(targets_np)):
                filename = Path(batch_paths[j]).name
                filename_without_ext = filename.rsplit(".", 1)[0]

                row = {
                    "image": filename_without_ext,
                    "true_label": int(targets_np[j]),
                    "pred_label": int(preds_np[j]),
                    "correct": int(targets_np[j]) == int(preds_np[j]),
                }

                for c in range(num_classes):
                    row[f"prob_class_{c}"] = float(probs_np[j, c])

                all_results.append(row)

    # === 5. 保存结果表格 ===
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    final_out_dir = Path(output_dir)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results).sort_values("image").reset_index(drop=True)
    csv_path = final_out_dir / f"{split}_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ {split.upper()} 测试完成！")
    print(f"   📄 原始结果 CSV: {csv_path}")

    # === 6. 计算并保存指标 ===
    try:
        txt_metrics, json_metrics = calculate_and_save_metrics(
            df=df,
            output_dir=output_dir,
            split=split,
            timestamp=timestamp,
            num_classes=num_classes
        )
        print(f"   📈 详细指标 TXT: {txt_metrics}")
        print(f"   📈 详细指标 JSON: {json_metrics}")
    except Exception as e:
        print(f"   ❌ 指标计算失败：{e}")
        import traceback
        traceback.print_exc()

    # === 7. 控制台简易统计 ===
    acc = df["correct"].mean()
    print(f"   ⚡ Accuracy: {acc:.4f} | 总样本数：{len(df)}\n")


def main():
    print("=" * 70)
    print("     EfficientNet 分类模型推理测试（兼容二分类 / 多分类）".center(70))
    print("=" * 70)

    best_ckpt = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs/260323_576_0.65_2class_noAug/epoch=33-val_acc_macro=0.0000.ckpt"

    # --- TJMU ---
    test_data_sir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU_labels.txt"
    res_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_TJMU_260323_noAug/"
    run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # --- BJH ---
    test_data_sir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH_labels.txt"
    res_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_BJH_260323_noAug/"
    run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # --- FXH_noALL ---
    test_data_sir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL_labels.txt"
    res_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_FXH_noALL_260323_noAug/"
    run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # --- train ---
    test_data_sir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train_labels.txt"
    res_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/train_260323_noAug/"
    run_test_on_split(split="train", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # --- val ---
    test_data_sir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val_labels.txt"
    res_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/val_260323_noAug/"
    run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)


if __name__ == "__main__":
    main()