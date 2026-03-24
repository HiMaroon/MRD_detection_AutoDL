'''
#有可解释性
# tools/calculate_fpr.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

DECIMALS = 4

# ==================== 辅助工具：自动适配列名 ====================
def get_p_col(df):
    """自动检测代表患者ID的列名（兼容 Patient ID 或 Patient_ID）"""
    candidates = ["Patient_ID", "Patient ID", "patient_id", "patient id"]
    for c in candidates:
        if c in df.columns:
            return c
    # 模糊搜索
    cols = [c for c in df.columns if 'patient' in c.lower() or 'id' in c.lower()]
    return cols[0] if cols else None

def cell_level_metrics(results_file_path, save_dir=None, keep_prob_analysis=True):
    print("=" * 60)
    print("细胞级：分类指标 + 错误置信度分析")
    print("=" * 60)

    df = pd.read_excel(results_file_path) if results_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(results_file_path)

    for col in ["true_label", "pred_label"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    p0, p1 = precisions
    r0, r1 = recalls
    f10, f11 = f1s
    s0, s1 = supports

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"总样本数: {len(y_true)} | TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Accuracy: {acc:.4f} | Precision(1): {p1:.4f} | Recall(1): {r1:.4f} | F1(1): {f11:.4f}")
    print(f"Specificity: {specificity:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    print("\n详细分类报告：")
    detailed_report = classification_report(
        y_true, y_pred,
        target_names=["正常细胞", "原始细胞"],
        digits=4,
        zero_division=0,
    )
    print(detailed_report)

    prob_col = next((c for c in ["prob_class_1", "pred_prob", "confidence", "prob"] if c in df.columns), None)

    if prob_col:
        print(f"\n检测到概率列: '{prob_col}'")
        threshold_est = df[df["pred_label"] == 1][prob_col].min()
        print(f"当前推理使用的阈值 ≈ {threshold_est:.4f}")

    if prob_col and keep_prob_analysis:
        print("\n" + "=" * 60)
        print("错误样本置信度分析")
        print("=" * 60)
        fp_df = df[(df["true_label"] == 0) & (df["pred_label"] == 1)]
        fn_df = df[(df["true_label"] == 1) & (df["pred_label"] == 0)]

        if len(fp_df) > 0:
            print(f"假阳性 (FP={len(fp_df)}): 平均置信度 {fp_df[prob_col].mean():.4f}")
        if len(fn_df) > 0:
            print(f"假阴性 (FN={len(fn_df)}): 平均置信度 {(1.0 - fn_df[prob_col]).mean():.4f}")

    if save_dir is None:
        save_dir = os.path.dirname(results_file_path) or "."
    out_path = os.path.join(save_dir, "细胞级_分类指标.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame({"指标": ["Accuracy", "FPR", "FNR"], "数值": [acc, fpr, fnr]}).to_excel(writer, sheet_name="总体", index=False)
    
    print(f"\n细胞级指标已保存 → {out_path}")

def patient_level_metrics(patient_file_path, save_dir=None):
    print("\n" + "=" * 60)
    print("患者级：分类指标分析")
    print("=" * 60)
    df = pd.read_excel(patient_file_path) if patient_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(patient_file_path)
    
    # 核心修复：自动适配患者ID列名
    pid_col = get_p_col(df)
    if not pid_col:
        raise ValueError("数据集中未找到 Patient ID 或 Patient_ID 列")
    
    need = ["true_positive", "true_negative", "pred_positive", "pred_negative"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    rows = []
    for _, r in df.iterrows():
        pid = r[pid_col]
        tp, tn, pp, pn = int(r["true_positive"]), int(r["true_negative"]), int(r["pred_positive"]), int(r["pred_negative"])
        fp, fn = max(0, pp - tp), max(0, pn - tn)
        total = tp + tn + fp + fn
        
        row = {
            "Patient ID": pid,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": round((tp + tn) / total, 4) if total > 0 else 0,
            "Recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
            "FPR": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        }
        if "type" in r: row["Type"] = r["type"]
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if save_dir is None: save_dir = os.path.dirname(patient_file_path) or "."
    out_path = os.path.join(save_dir, "患者级_分类指标.xlsx")
    metrics_df.to_excel(out_path, index=False)
    print(f"✅ 患者级指标分析完成，保存至：{out_path}")
    return metrics_df

def calculate_patient_level_fpr(patient_analysis_file_path):
    print("\n患者级别假阳率(FPR)统计")
    df = pd.read_excel(patient_analysis_file_path) if patient_analysis_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(patient_analysis_file_path)
    
    pid_col = get_p_col(df)
    required = ["true_positive", "true_negative", "pred_positive"]
    
    results = []
    for _, row in df.iterrows():
        fp = max(0, row["pred_positive"] - row["true_positive"])
        results.append({
            "Patient ID": row[pid_col],
            "false_positive": fp,
            "true_negative": row["true_negative"],
            "fpr": fp / row["true_negative"] if row["true_negative"] > 0 else 0
        })

    fpr_df = pd.DataFrame(results)
    overall_fpr = fpr_df["false_positive"].sum() / fpr_df["true_negative"].sum() if fpr_df["true_negative"].sum() > 0 else 0
    print(f"📊 总体假阳率: {overall_fpr:.4f}")
    
    out_file = patient_analysis_file_path.replace(".xlsx", "_patient_fpr.xlsx")
    fpr_df.to_excel(out_file, index=False)
    print(f"✅ 患者级 FPR 已保存 → {out_file}")

def main():
    # 路径配置 (仅修改这里)
    root_path = "/root/autodl-tmp/results/yolo_with_o_features"
    
    # 自动定位刚才生成的预测结果和患者汇总文件
    # 1. 找 CSV (test.py 生成的)
    files = [f for f in os.listdir(root_path) if f.startswith("val_results") and f.endswith(".csv")]
    if not files:
        # 如果没有csv，找xlsx
        files = [f for f in os.listdir(root_path) if f.startswith("val_results") and f.endswith(".xlsx")]
    
    if not files:
        print(f"❌ 错误: 未找到 val_results 文件在 {root_path}")
        return
        
    files.sort(reverse=True)
    test_results_path = os.path.join(root_path, files[0])
    
    # 2. 找刚才 compute_patient 生成的分析表
    patient_analysis_path = os.path.join(root_path, "patient_analysis_v_o_f.xlsx")

    if os.path.exists(test_results_path):
        cell_level_metrics(test_results_path)
    else:
        print(f"❌ 细胞结果文件不存在: {test_results_path}")

    if os.path.exists(patient_analysis_path):
        patient_level_metrics(patient_analysis_path)
        calculate_patient_level_fpr(patient_analysis_path)
    else:
        print(f"❌ 患者汇总文件不存在: {patient_analysis_path}")

if __name__ == "__main__":
    main()
'''



#无可解释性
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

DECIMALS = 4

# ==================== 辅助工具：自动适配列名 ====================
def get_p_col(df):
    """自动检测代表患者ID的列名（兼容 Patient ID 或 Patient_ID）"""
    candidates = ["Patient_ID", "Patient ID", "patient_id", "patient id"]
    for c in candidates:
        if c in df.columns:
            return c
    # 模糊搜索
    cols = [c for c in df.columns if 'patient' in c.lower() or 'id' in c.lower()]
    return cols[0] if cols else None

def cell_level_metrics(results_file_path, save_dir=None, keep_prob_analysis=True):
    print("=" * 60)
    print("细胞级：分类指标 + 错误置信度分析")
    print("=" * 60)

    df = pd.read_excel(results_file_path) if results_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(results_file_path)

    for col in ["true_label", "pred_label"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    p0, p1 = precisions
    r0, r1 = recalls
    f10, f11 = f1s
    s0, s1 = supports

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"总样本数: {len(y_true)} | TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Accuracy: {acc:.4f} | Precision(1): {p1:.4f} | Recall(1): {r1:.4f} | F1(1): {f11:.4f}")
    print(f"Specificity: {specificity:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    print("\n详细分类报告：")
    detailed_report = classification_report(
        y_true, y_pred,
        target_names=["正常细胞", "原始细胞"],
        digits=4,
        zero_division=0,
    )
    print(detailed_report)

    prob_col = next((c for c in ["prob_class_1", "pred_prob", "confidence", "prob"] if c in df.columns), None)

    if prob_col:
        print(f"\n检测到概率列: '{prob_col}'")
        threshold_est = df[df["pred_label"] == 1][prob_col].min()
        print(f"当前推理使用的阈值 ≈ {threshold_est:.4f}")

    if prob_col and keep_prob_analysis:
        print("\n" + "=" * 60)
        print("错误样本置信度分析")
        print("=" * 60)
        fp_df = df[(df["true_label"] == 0) & (df["pred_label"] == 1)]
        fn_df = df[(df["true_label"] == 1) & (df["pred_label"] == 0)]

        if len(fp_df) > 0:
            print(f"假阳性 (FP={len(fp_df)}): 平均置信度 {fp_df[prob_col].mean():.4f}")
        if len(fn_df) > 0:
            print(f"假阴性 (FN={len(fn_df)}): 平均置信度 {(1.0 - fn_df[prob_col]).mean():.4f}")

    if save_dir is None:
        save_dir = os.path.dirname(results_file_path) or "."
    out_path = os.path.join(save_dir, "细胞级_分类指标.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame({"指标": ["Accuracy", "FPR", "FNR"], "数值": [acc, fpr, fnr]}).to_excel(writer, sheet_name="总体", index=False)
    
    print(f"\n细胞级指标已保存 → {out_path}")

def patient_level_metrics(patient_file_path, save_dir=None):
    print("\n" + "=" * 60)
    print("患者级：分类指标分析")
    print("=" * 60)
    df = pd.read_excel(patient_file_path) if patient_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(patient_file_path)
    
    # 核心修复：自动适配患者ID列名
    pid_col = get_p_col(df)
    if not pid_col:
        raise ValueError("数据集中未找到 Patient ID 或 Patient_ID 列")
    
    need = ["true_positive", "true_negative", "pred_positive", "pred_negative"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    rows = []
    for _, r in df.iterrows():
        pid = r[pid_col]
        tp, tn, pp, pn = int(r["true_positive"]), int(r["true_negative"]), int(r["pred_positive"]), int(r["pred_negative"])
        fp, fn = max(0, pp - tp), max(0, pn - tn)
        total = tp + tn + fp + fn
        
        row = {
            "Patient ID": pid,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": round((tp + tn) / total, 4) if total > 0 else 0,
            "Recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
            "FPR": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        }
        if "type" in r: row["Type"] = r["type"]
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if save_dir is None: save_dir = os.path.dirname(patient_file_path) or "."
    out_path = os.path.join(save_dir, "患者级_分类指标.xlsx")
    metrics_df.to_excel(out_path, index=False)
    print(f"✅ 患者级指标分析完成，保存至：{out_path}")
    return metrics_df

def calculate_patient_level_fpr(patient_analysis_file_path):
    print("\n患者级别假阳率(FPR)统计")
    df = pd.read_excel(patient_analysis_file_path) if patient_analysis_file_path.endswith((".xlsx", ".xls")) else pd.read_csv(patient_analysis_file_path)
    
    pid_col = get_p_col(df)
    required = ["true_positive", "true_negative", "pred_positive"]
    
    results = []
    for _, row in df.iterrows():
        fp = max(0, row["pred_positive"] - row["true_positive"])
        results.append({
            "Patient ID": row[pid_col],
            "false_positive": fp,
            "true_negative": row["true_negative"],
            "fpr": fp / row["true_negative"] if row["true_negative"] > 0 else 0
        })

    fpr_df = pd.DataFrame(results)
    overall_fpr = fpr_df["false_positive"].sum() / fpr_df["true_negative"].sum() if fpr_df["true_negative"].sum() > 0 else 0
    print(f"📊 总体假阳率: {overall_fpr:.4f}")
    
    out_file = patient_analysis_file_path.replace(".xlsx", "_patient_fpr.xlsx")
    fpr_df.to_excel(out_file, index=False)
    print(f"✅ 患者级 FPR 已保存 → {out_file}")

def main():
    # 路径配置
    root_path = "/root/autodl-tmp/results/cellpose_with_modified_data"  #/root/autodl-tmp/results/yolo_with_modified_data
    # 自动定位刚才生成的预测结果和患者汇总文件
    test_results_path = os.path.join(root_path, "val_results_20260108-113716.xlsx") #无预训练 m:val_results_20251224-152112.xlsx o:val_results_20251224-150009.xlsx
    patient_analysis_path = os.path.join(root_path, "patient_analysis_v_m.xlsx")

    if os.path.exists(test_results_path):
        cell_level_metrics(test_results_path)
    else:
        print(f"❌ 细胞结果文件不存在: {test_results_path}")

    if os.path.exists(patient_analysis_path):
        patient_level_metrics(patient_analysis_path)
        calculate_patient_level_fpr(patient_analysis_path)
    else:
        print(f"❌ 患者汇总文件不存在: {patient_analysis_path}")

if __name__ == "__main__":
    main()
