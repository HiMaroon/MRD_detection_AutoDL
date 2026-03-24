# tools/test.py
import os
import sys
import time
import pandas as pd
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

# 引入 sklearn 用于计算指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

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

def calculate_and_save_metrics(df, output_dir, split, timestamp):
    """
    计算分类指标并保存到 TXT 文件（包含每个类别的单独指标）
    """
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    y_prob = df['prob_class_1'].values
    
    metrics = {}
    report_text = []
    
    # 1. 基础指标
    acc = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = float(acc)
    
    # 2. 使用 classification_report 获取每个类别的详细指标
    unique_labels = np.unique(y_true)
    num_classes = len(unique_labels)
    
    report_text.append("=" * 70)
    report_text.append(f"📊 {split.upper()} SET EVALUATION METRICS")
    report_text.append("=" * 70)
    report_text.append(f"Total Samples: {len(df)}")
    report_text.append(f"Class Distribution (True): {dict(zip(*np.unique(y_true, return_counts=True)))}")
    report_text.append(f"Class Distribution (Pred): {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    report_text.append("-" * 70)
    
    try:
        # 获取每个类别的指标
        report_dict = classification_report(
            y_true, y_pred, 
            output_dict=True, 
            zero_division=0
        )
        
        report_text.append("\n📋 PER-CLASS METRICS:")
        report_text.append("-" * 70)
        
        # 遍历每个类别 (0 和 1)
        for cls in sorted([int(k) for k in report_dict.keys() if k.isdigit()]):
            cls_key = str(cls)
            cls_metrics = report_dict[cls_key]
            
            prec = cls_metrics['precision']
            rec = cls_metrics['recall']
            f1 = cls_metrics['f1-score']
            support = cls_metrics['support']
            
            metrics[f'class_{cls}_precision'] = float(prec)
            metrics[f'class_{cls}_recall'] = float(rec)
            metrics[f'class_{cls}_f1'] = float(f1)
            metrics[f'class_{cls}_support'] = int(support)
            
            report_text.append(f"Class {cls}:")
            report_text.append(f"  Precision:  {prec:.4f}")
            report_text.append(f"  Recall:     {rec:.4f}")
            report_text.append(f"  F1-Score:   {f1:.4f}")
            report_text.append(f"  Support:    {support}")
            report_text.append("")
        
        # 整体指标 (Macro/Weighted)
        report_text.append("📋 OVERALL METRICS:")
        report_text.append("-" * 70)
        
        # Macro 平均
        macro_prec = report_dict['macro avg']['precision']
        macro_rec = report_dict['macro avg']['recall']
        macro_f1 = report_dict['macro avg']['f1-score']
        metrics['macro_precision'] = float(macro_prec)
        metrics['macro_recall'] = float(macro_rec)
        metrics['macro_f1'] = float(macro_f1)
        
        report_text.append(f"Macro Avg - Precision: {macro_prec:.4f}, Recall: {macro_rec:.4f}, F1: {macro_f1:.4f}")
        
        # Weighted 平均
        weighted_prec = report_dict['weighted avg']['precision']
        weighted_rec = report_dict['weighted avg']['recall']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        metrics['weighted_precision'] = float(weighted_prec)
        metrics['weighted_recall'] = float(weighted_rec)
        metrics['weighted_f1'] = float(weighted_f1)
        
        report_text.append(f"Weighted Avg - Precision: {weighted_prec:.4f}, Recall: {weighted_rec:.4f}, F1: {weighted_f1:.4f}")
        
    except Exception as e:
        report_text.append(f"⚠️ Classification Report 计算警告：{e}")
    
    # 3. AUC 指标
    report_text.append("")
    report_text.append("📋 AUC METRICS:")
    report_text.append("-" * 70)
    
    try:
        if num_classes > 1:
            auc_roc = roc_auc_score(y_true, y_prob)
            metrics['auc_roc'] = float(auc_roc)
            report_text.append(f"ROC-AUC:       {auc_roc:.4f}")
        else:
            metrics['auc_roc'] = 0.5
            report_text.append("ROC-AUC:       N/A (单类别)")
    except Exception as e:
        metrics['auc_roc'] = -1.0
        report_text.append(f"ROC-AUC:       计算错误 ({e})")
    
    # 4. 混淆矩阵
    report_text.append("")
    report_text.append("📋 CONFUSION MATRIX:")
    report_text.append("-" * 70)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    report_text.append("                Predicted")
    report_text.append("                Class 0     Class 1")
    report_text.append(f"Actual Class 0    {cm[0][0]:>6}      {cm[0][1]:>6}")
    report_text.append(f"Actual Class 1    {cm[1][0]:>6}      {cm[1][1]:>6}")
    
    # 计算混淆矩阵衍生指标
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        report_text.append("")
        report_text.append("📋 CONFUSION MATRIX DETAILS:")
        report_text.append(f"  TN (真阴性): {tn}")
        report_text.append(f"  FP (假阳性): {fp}")
        report_text.append(f"  FN (假阴性): {fn}")
        report_text.append(f"  TP (真阳性): {tp}")
    
    report_text.append("=" * 70)
    
    # 5. 保存文件
    final_out_dir = Path(output_dir)
    final_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 TXT (人类可读)
    txt_path = final_out_dir / f"{split}_metrics_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_text))
    
    return txt_path

def run_test_on_split(split: str = "val", ckpt_path: str = None, test_data_sir: str = None, output_dir: str = None):
    assert split in ["train", "val"]

    # === 1. 验证 Checkpoint ===
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到指定的 checkpoint: {ckpt_path}")
    print(f"🚀 使用 checkpoint: {ckpt_path}")

    # === 2. 加载模型 ===
    core = LitSingleCell(model_cfg, num_classes=data_cfg["num_classes"])
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
    print(f"📊 [{split.upper()} SET] 样本总数：{len(dataset)} | BatchSize: {loader.batch_size}")

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
            
            result = model.core.validation_step((x, y))
            
            if isinstance(result, dict) and "probs" in result:
                probs_1 = result["probs"]
            else:
                logits = model.core(x)
                probs_1 = torch.softmax(logits, dim=1)[:, 1]

            probs_0 = 1.0 - probs_1
            preds = (probs_1 >= 0.5).long() 
            
            probs_0 = probs_0.cpu().tolist()
            probs_1 = probs_1.cpu().tolist()
            preds = preds.cpu().tolist()
            targets = y.cpu().tolist()
            
            for j in range(len(targets)):
                filename = Path(batch_paths[j]).name
                filename_without_ext = filename.rsplit('.', 1)[0]
                
                all_results.append({
                    "image": filename_without_ext,
                    "true_label": int(targets[j]),
                    "pred_label": int(preds[j]),
                    "prob_class_0": float(probs_0[j]),
                    "prob_class_1": float(probs_1[j]),
                    "correct": int(targets[j]) == int(preds[j])
                })

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
        txt_metrics = calculate_and_save_metrics(df, output_dir, split, timestamp)
        print(f"   📈 详细指标 TXT: {txt_metrics}")
    except Exception as e:
        print(f"   ❌ 指标计算失败：{e}")
        import traceback
        traceback.print_exc()

    # === 7. 控制台简易统计 ===
    acc = df["correct"].mean()
    print(f"   ⚡ Accuracy: {acc:.4f} | 总样本数：{len(df)}\n")

def main():
    print("=" * 70)
    print("     EfficientNet 分类模型推理测试 ( 预训练版 )".center(70))
    print("=" * 70)

    best_ckpt = "/root/autodl-tmp/projects/myq/SingleCellProject/outputs/260316cellseg_2class_576/epoch=48-val_acc_macro=0.9103.ckpt"  
    
    # # --- FXH ---
    # test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_FXH_labels.txt"
    # res_dir = "/root/autodl-tmp/results/External_validation_FXH/"
    # run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # # --- BJH ---
    # test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_BJH_labels.txt"
    # res_dir = "/root/autodl-tmp/results/External_validation_BJH/"
    # run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # # --- BEPH ---
    # test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_BEPH_labels.txt"
    # res_dir = "/root/autodl-tmp/results/External_validation_BEPH/"
    # run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # # --- TJMU ---
    # test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_TJMU_labels.txt"
    # res_dir = "/root/autodl-tmp/results/External_validation_TJMU/"
    # run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # # --- FXH_noALL ---
    # test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_FXH_noALL_labels.txt"
    # res_dir = "/root/autodl-tmp/results/External_validation_FXH_noALL/"
    # run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

    # --- TJMU ---
    test_data_sir="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_TJMU_transfer_labels.txt"
    res_dir = "/root/autodl-tmp/results/External_validation_TJMU_transfer/"
    run_test_on_split(split="val", ckpt_path=best_ckpt, test_data_sir=test_data_sir, output_dir=res_dir)

if __name__ == "__main__":
    main()