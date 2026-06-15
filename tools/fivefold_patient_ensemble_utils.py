from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_PATH = "/root/autodl-tmp/projects/myq/SingleCellProject/tools/MSYH.TTC"
FONT_NAME = "MSYH.TTC"
POSITIVE_CLASS = 1

# N/M-only primitive-cell definition:
# value == 1 is primitive positive; value == 0/2 or missing is negative.
CELL_TYPE_TO_BIG = {
    "V": 0,
    "0": 0,
    "N": 1,
    "N1": 2,
    "M": 1,
    "M1": 2,
    "R": 2,
    "R1": 2,
    "J": 2,
    "J1": 2,
    "N0": 2,
    "N2": 2,
    "N3": 2,
    "N4": 2,
    "N5": 2,
    "E": 2,
    "B": 2,
    "E1": 2,
    "B1": 2,
    "M0": 2,
    "M2": 2,
    "R2": 2,
    "R3": 2,
    "J2": 2,
    "J3": 2,
    "J4": 2,
    "P": 2,
    "P1": 2,
    "P2": 2,
    "P3": 2,
    "L": 2,
    "L1": 2,
    "L2": 2,
    "L3": 2,
    "L4": 2,
}


def write_dataframe_outputs(df: pd.DataFrame, csv_path: Path, xlsx_path: Optional[Path] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if xlsx_path is None:
        return
    try:
        df.to_excel(xlsx_path, index=False)
    except ImportError as e:
        print(f"⚠️ 缺少 Excel 写入依赖，已仅保存 CSV: {csv_path} | {e}")

def set_chinese_font() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [FONT_PATH, os.path.join(current_dir, FONT_NAME), os.path.join(os.getcwd(), FONT_NAME)]
    selected_font = next((p for p in candidates if p and os.path.exists(p)), None)
    if selected_font:
        try:
            fm.fontManager.addfont(selected_font)
            font_prop = fm.FontProperties(fname=selected_font)
            plt.rcParams["font.family"] = font_prop.get_name()
            print(f"✅ 已设置中文字体: {selected_font}")
        except Exception as e:
            print(f"⚠️ 字体注册异常: {e}")
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    else:
        print(f"⚠️ 未找到字体 {FONT_NAME}，使用系统回退字体")
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def discover_fold_csvs(root: Path, split_name: str) -> List[Path]:
    return sorted(root.glob(f"fold*/eval/{split_name}/val_results.csv"))


def parse_image_info(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(str(image_name)).stem
    prefix = stem.split("_")[0]
    parts = prefix.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}", parts[2]
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}", None
    return None, None


def parse_cell_type(image_name: str) -> Optional[str]:
    if pd.isna(image_name):
        return None
    stem = Path(str(image_name).strip()).stem
    if "_" not in stem:
        return None
    return stem.split("_")[-1].strip().upper()


def map_cell_type_to_binary(cell_type: Optional[str]) -> int:
    if pd.isna(cell_type) or cell_type is None:
        return 0
    value = CELL_TYPE_TO_BIG.get(str(cell_type).strip().upper(), 0)
    return 1 if value == 1 else 0


def normalize_patient_type(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    if x in {"HC", "HD", "NORMAL", "HEALTHY"}:
        return "HC"
    if x == "AML":
        return "AML"
    return x


def load_patient_info(patient_info_xlsx: Optional[Path]) -> Optional[pd.DataFrame]:
    if patient_info_xlsx is None or str(patient_info_xlsx).strip() == "":
        return None
    if patient_info_xlsx.is_dir():
        print(f"⚠️ 患者信息路径是目录而不是 xlsx，继续做无类型患者分析: {patient_info_xlsx}")
        return None
    if not patient_info_xlsx.exists():
        print(f"⚠️ 患者信息文件不存在，继续做无类型患者分析: {patient_info_xlsx}")
        return None
    df = pd.read_excel(patient_info_xlsx)
    print(f"📖 已读取患者信息: {patient_info_xlsx}")
    return df


def merge_fold_probabilities(
    fold_csvs: Sequence[Path],
    *,
    keep_fold_prob_cols: bool = False,
) -> pd.DataFrame:
    if len(fold_csvs) == 0:
        raise ValueError("fold_csvs 为空")

    dfs = [pd.read_csv(p) for p in fold_csvs]
    prob_cols = [c for c in dfs[0].columns if c.startswith("prob_class_")]
    if "image" not in dfs[0].columns:
        raise ValueError(f"{fold_csvs[0]} 缺少 image 列")
    if "prob_class_1" not in prob_cols:
        raise ValueError("需要二分类概率列 prob_class_1 才能做阈值扫描/概率均值分类")

    base_cols = ["image"]
    if "true_label" in dfs[0].columns:
        base_cols.append("true_label")
    merged = dfs[0][base_cols].copy()

    for idx, (df, csv_path) in enumerate(zip(dfs, fold_csvs), start=1):
        missing = [c for c in ["image", *prob_cols] if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少列: {missing}")
        sub = df[["image", *prob_cols]].copy()
        sub = sub.rename(columns={c: f"{c}_f{idx}" for c in prob_cols})
        merged = merged.merge(sub, on="image", how="inner")

    for col in prob_cols:
        fold_cols = [f"{col}_f{i}" for i in range(1, len(dfs) + 1)]
        merged[col] = merged[fold_cols].mean(axis=1)
        merged[f"{col}_std"] = merged[fold_cols].std(axis=1, ddof=0)

    merged["fold_count"] = len(dfs)
    merged["ensemble_prob_class_1"] = merged["prob_class_1"]
    keep_cols = ["image"]
    if "true_label" in merged.columns:
        keep_cols.append("true_label")
    keep_cols.extend(["fold_count", *prob_cols, "ensemble_prob_class_1", "prob_class_1_std"])
    if keep_fold_prob_cols:
        for col in prob_cols:
            keep_cols.extend([f"{col}_f{i}" for i in range(1, len(dfs) + 1)])
    return merged[keep_cols].sort_values("image").reset_index(drop=True)


def add_cell_predictions(merged_probs: pd.DataFrame, cell_threshold: float) -> pd.DataFrame:
    df = merged_probs.copy()
    df["pred_label"] = (df["prob_class_1"].astype(float) >= float(cell_threshold)).astype(int)
    if "true_label" in df.columns:
        df["correct"] = df["true_label"].astype(int) == df["pred_label"].astype(int)
    return df


def attach_patient_fields(cell_df: pd.DataFrame) -> pd.DataFrame:
    df = cell_df.copy()
    parsed = df["image"].apply(parse_image_info)
    df["patient_id"] = parsed.apply(lambda x: x[0])
    df["smear_id"] = parsed.apply(lambda x: x[1])
    df["cell_type"] = df["image"].apply(parse_cell_type)
    df["mapped_label"] = df["cell_type"].apply(map_cell_type_to_binary)
    df = df[df["patient_id"].notna()].copy()
    df["actual_positive"] = (df["mapped_label"].astype(int) == POSITIVE_CLASS).astype(int)
    return df


def build_patient_summary(
    cell_df: pd.DataFrame,
    patient_info_df: Optional[pd.DataFrame] = None,
    *,
    patient_prob_threshold: float = 0.5,
    patient_hard_ratio_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = attach_patient_fields(cell_df)
    if "pred_label" not in df.columns:
        raise ValueError("cell_df 缺少 pred_label 列；请先按细胞阈值生成硬标签")
    if "prob_class_1" not in df.columns:
        raise ValueError("cell_df 缺少 prob_class_1 列")

    df["pred_positive"] = (df["pred_label"].astype(int) == POSITIVE_CLASS).astype(int)
    df["prob_class_1"] = pd.to_numeric(df["prob_class_1"], errors="coerce").fillna(0.0)
    df["is_correct_by_mapped_label"] = (df["mapped_label"].astype(int) == df["pred_label"].astype(int)).astype(int)

    patient_summary = (
        df.groupby("patient_id")
        .agg(
            n_cells=("image", "count"),
            n_smears=("smear_id", "nunique"),
            actual_ratio=("actual_positive", "mean"),
            hard_predicted_ratio=("pred_positive", "mean"),
            predicted_ratio=("pred_positive", "mean"),
            mean_prob_class_1=("prob_class_1", "mean"),
            median_prob_class_1=("prob_class_1", "median"),
            p90_prob_class_1=("prob_class_1", lambda x: float(np.quantile(x, 0.90))),
            p95_prob_class_1=("prob_class_1", lambda x: float(np.quantile(x, 0.95))),
            accuracy=("is_correct_by_mapped_label", "mean"),
        )
        .reset_index()
    )
    patient_summary["actual_patient_label"] = (patient_summary["actual_ratio"] > 0).astype(int)
    patient_summary["pred_label_by_hard_ratio"] = (
        patient_summary["hard_predicted_ratio"] >= float(patient_hard_ratio_threshold)
    ).astype(int)
    patient_summary["pred_label_by_mean_prob"] = (
        patient_summary["mean_prob_class_1"] >= float(patient_prob_threshold)
    ).astype(int)

    if patient_info_df is not None and len(patient_info_df) > 0:
        info = patient_info_df.copy()
        if "正式编号" in info.columns:
            info["正式编号"] = info["正式编号"].astype(str).str.strip()
        if "患者大类型" in info.columns:
            info["患者大类型"] = info["患者大类型"].apply(normalize_patient_type)
        if {"正式编号", "患者大类型"}.issubset(info.columns):
            patient_summary = patient_summary.merge(
                info[["正式编号", "患者大类型"]].drop_duplicates(),
                left_on="patient_id",
                right_on="正式编号",
                how="left",
            )
            patient_summary.rename(columns={"患者大类型": "type"}, inplace=True)
            patient_summary.drop(columns=["正式编号"], inplace=True)
        else:
            patient_summary["type"] = np.nan
    else:
        patient_summary["type"] = np.nan

    return patient_summary, df


def patient_type_to_binary(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda x: 0 if normalize_patient_type(x) == "HC" else (1 if normalize_patient_type(x) == "AML" else np.nan)
    )


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def _roc_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.array([np.inf, *sorted(np.unique(scores), reverse=True), -np.inf], dtype=float)
    fprs = []
    tprs = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = _binary_counts(y_true, pred)
        fprs.append(fp / max(fp + tn, 1))
        tprs.append(tp / max(tp + fn, 1))
    return np.asarray(fprs, dtype=float), np.asarray(tprs, dtype=float), thresholds


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = _roc_points(y_true, scores)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def evaluate_patient_predictions(
    patient_summary: pd.DataFrame,
    *,
    pred_col: str,
    score_col: Optional[str] = None,
) -> dict:
    result = {"n_patients": int(len(patient_summary))}
    if "type" not in patient_summary.columns:
        return result
    df = patient_summary.copy()
    df["type_binary"] = patient_type_to_binary(df["type"])
    df = df[df["type_binary"].notna()].copy()
    if len(df) == 0:
        return result
    y_true = df["type_binary"].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    result.update(
        {
            "n_labeled_patients": int(len(df)),
            "accuracy": float((y_true == y_pred).mean()),
        }
    )
    tn, fp, fn, tp = _binary_counts(y_true, y_pred)
    result.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "specificity_hc": float(tn / max(tn + fp, 1)),
            "sensitivity_aml": float(tp / max(tp + fn, 1)),
        }
    )
    if score_col and score_col in df.columns and df["type_binary"].nunique() == 2:
        auc = _binary_auc(y_true, df[score_col].astype(float).to_numpy())
        if auc is not None:
            result["patient_auc"] = auc
    return result


def best_patient_threshold_by_roc(patient_summary: pd.DataFrame, score_col: str) -> Optional[float]:
    if "type" not in patient_summary.columns or score_col not in patient_summary.columns:
        return None
    df = patient_summary.copy()
    df["type_binary"] = patient_type_to_binary(df["type"])
    df = df[df["type_binary"].notna()].copy()
    if len(df) == 0 or df["type_binary"].nunique() < 2:
        return None
    fpr, tpr, thresholds = _roc_points(df["type_binary"].astype(int).to_numpy(), df[score_col].astype(float).to_numpy())
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def plot_patient_ratios(
    patient_summary: pd.DataFrame,
    save_path: Path,
    *,
    ratio_col: str,
    title: str,
    label: str,
) -> None:
    if len(patient_summary) == 0:
        return
    df = patient_summary.copy()
    df["sort_key"] = df[ratio_col]
    if "type" in df.columns:
        df["type_sort"] = df["type"].map({"HC": 0, "AML": 1}).fillna(2)
    else:
        df["type_sort"] = 2
    df = df.sort_values(["type_sort", "sort_key", "patient_id"]).reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.38
    fig_w = max(14, len(df) * 0.36)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    bars1 = ax.bar(x - width / 2, df["actual_ratio"], width, label="实际N/M比例", color="lightcoral", edgecolor="red")
    bars2 = ax.bar(x + width / 2, df[ratio_col], width, label=label, color="lightblue", edgecolor="blue")

    for bars in (bars1, bars2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    labels = [f"{pid}\n{t}" if pd.notna(t) else str(pid) for pid, t in zip(df["patient_id"], df.get("type", pd.Series([np.nan] * len(df))))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, min(1.0, max(float(df["actual_ratio"].max()), float(df[ratio_col].max())) + 0.15))
    ax.set_ylabel("比例")
    ax.set_xlabel("患者")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    for lbl, t in zip(ax.get_xticklabels(), df.get("type", pd.Series([np.nan] * len(df)))):
        if t == "AML":
            lbl.set_color("red")
        elif t == "HC":
            lbl.set_color("blue")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_thresholds(thresholds: Optional[Iterable[str]], start: float, end: float, step: float) -> List[float]:
    if thresholds:
        values: List[float] = []
        for item in thresholds:
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    values.append(float(part))
        return sorted(set(round(v, 6) for v in values))
    if step <= 0:
        raise ValueError("step 必须 > 0")
    values = []
    cur = start
    while cur <= end + 1e-12:
        values.append(round(cur, 6))
        cur += step
    return values
