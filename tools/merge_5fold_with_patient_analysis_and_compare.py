import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.test import calculate_and_save_metrics


# =========================
# 全局配置
# =========================
font_path = "/root/autodl-tmp/projects/myq/SingleCellProject/tools/MSYH.TTC"
FONT_NAME = "MSYH.TTC"
POSITIVE_CLASS = 1

# 规则：只有 value == 1 的算正类；value == 0/2 或不在字典中都算 0
cell_dict_big = {
    "V": 0, "0": 0,
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2,
}


# =========================
# 基础工具
# =========================
def ensure_parent_dir(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)


def set_chinese_font():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        font_path,
        os.path.join(current_dir, FONT_NAME),
        os.path.join(os.getcwd(), FONT_NAME),
    ]

    selected_font = None
    for path in candidates:
        if path and os.path.exists(path):
            selected_font = path
            break

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


# =========================
# 5-fold 融合部分
# =========================
def discover_fold_csvs(root: Path, split_name: str) -> List[Path]:
    return sorted(root.glob(f"fold*/eval/{split_name}/val_results.csv"))


def merge_with_probs(dfs: List[pd.DataFrame], binary_threshold: float) -> pd.DataFrame:
    prob_cols = [c for c in dfs[0].columns if c.startswith("prob_class_")]
    num_classes = len(prob_cols)

    merged = dfs[0][["image"]].copy()
    if "true_label" in dfs[0].columns:
        merged["true_label"] = dfs[0]["true_label"].values

    for i, df in enumerate(dfs, start=1):
        sub = df[["image"] + prob_cols].copy()
        sub = sub.rename(columns={c: f"{c}_f{i}" for c in prob_cols})
        merged = merged.merge(sub, on="image", how="inner")

    for c in prob_cols:
        fold_cols = [f"{c}_f{i}" for i in range(1, len(dfs) + 1)]
        merged[c] = merged[fold_cols].mean(axis=1)

    if num_classes == 2:
        merged["pred_label"] = (merged["prob_class_1"] >= binary_threshold).astype(int)
    else:
        merged["pred_label"] = merged[prob_cols].values.argmax(axis=1)

    if "true_label" in merged.columns:
        merged["correct"] = (
            merged["true_label"].astype(int) == merged["pred_label"].astype(int)
        )

    keep_cols = ["image"]
    if "true_label" in merged.columns:
        keep_cols += ["true_label"]
    keep_cols += ["pred_label"]
    if "correct" in merged.columns:
        keep_cols += ["correct"]
    keep_cols += prob_cols
    return merged[keep_cols].sort_values("image").reset_index(drop=True)


def merge_with_vote(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    merged = dfs[0][["image"]].copy()
    if "true_label" in dfs[0].columns:
        merged["true_label"] = dfs[0]["true_label"].values

    pred_cols = []
    for i, df in enumerate(dfs, start=1):
        col = f"pred_label_f{i}"
        pred_cols.append(col)
        sub = df[["image", "pred_label"]].rename(columns={"pred_label": col})
        merged = merged.merge(sub, on="image", how="inner")

    votes = merged[pred_cols].to_numpy(dtype=int)
    majority = []
    for row in votes:
        labels, counts = np.unique(row, return_counts=True)
        majority.append(int(labels[np.argmax(counts)]))

    merged["pred_label"] = majority
    if "true_label" in merged.columns:
        merged["correct"] = (
            merged["true_label"].astype(int) == merged["pred_label"].astype(int)
        )

    keep_cols = ["image"]
    if "true_label" in merged.columns:
        keep_cols += ["true_label"]
    keep_cols += ["pred_label"]
    if "correct" in merged.columns:
        keep_cols += ["correct"]
    return merged[keep_cols].sort_values("image").reset_index(drop=True)


# =========================
# 患者级分析工具函数
# =========================
def normalize_patient_type(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()
    if x in ["HC", "HD", "NORMAL", "HEALTHY"]:
        return "HC"
    if x in ["AML"]:
        return "AML"
    return x


def parse_image_info(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    例：
    PKUPH-106-10_000_P2 -> patient_id=PKUPH-106, smear_id=10
    FXH-1_000_P2        -> patient_id=FXH-1,  smear_id=None
    """
    image_name = str(image_name)
    stem = Path(image_name).stem
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

    image_name = str(image_name).strip()
    stem = Path(image_name).stem
    if "_" not in stem:
        return None

    return stem.split("_")[-1].strip().upper()


def map_cell_type_to_binary(cell_type: Optional[str]) -> int:
    if pd.isna(cell_type) or cell_type is None:
        return 0
    v = cell_dict_big.get(str(cell_type).strip().upper(), 0)
    return 1 if v == 1 else 0


def build_patient_summary(cell_df: pd.DataFrame, patient_info_df: Optional[pd.DataFrame] = None):
    """
    患者级统计依赖细胞级结果：image / pred_label / prob_class_*。
    actual_ratio 不依赖 true_label，而是从 image 名中的 cell_type 映射而来。
    """
    cell_df = cell_df.copy()

    parsed = cell_df["image"].apply(parse_image_info)
    cell_df["patient_id"] = parsed.apply(lambda x: x[0])
    cell_df["smear_id"] = parsed.apply(lambda x: x[1])
    cell_df["cell_type"] = cell_df["image"].apply(parse_cell_type)
    cell_df["mapped_label"] = cell_df["cell_type"].apply(map_cell_type_to_binary)

    bad_rows = cell_df["patient_id"].isna().sum()
    if bad_rows > 0:
        print(f"⚠️ 有 {bad_rows} 条记录无法解析 patient_id，已忽略")
        cell_df = cell_df[cell_df["patient_id"].notna()].copy()

    cell_df["actual_positive"] = (cell_df["mapped_label"] == POSITIVE_CLASS).astype(int)
    cell_df["pred_positive"] = (cell_df["pred_label"].astype(int) == POSITIVE_CLASS).astype(int)
    cell_df["is_correct"] = (cell_df["mapped_label"].astype(int) == cell_df["pred_label"].astype(int)).astype(int)

    if "prob_class_1" in cell_df.columns:
        patient_summary = (
            cell_df.groupby("patient_id")
            .agg(
                n_cells=("image", "count"),
                n_smears=("smear_id", "nunique"),
                actual_ratio=("actual_positive", "mean"),
                predicted_ratio=("pred_positive", "mean"),
                mean_prob_class_1=("prob_class_1", "mean"),
                accuracy=("is_correct", "mean"),
            )
            .reset_index()
        )
    else:
        patient_summary = (
            cell_df.groupby("patient_id")
            .agg(
                n_cells=("image", "count"),
                n_smears=("smear_id", "nunique"),
                actual_ratio=("actual_positive", "mean"),
                predicted_ratio=("pred_positive", "mean"),
                mean_prob_class_1=("pred_positive", "mean"),
                accuracy=("is_correct", "mean"),
            )
            .reset_index()
        )

    if patient_info_df is not None and len(patient_info_df) > 0:
        patient_info_df = patient_info_df.copy()
        if "正式编号" in patient_info_df.columns:
            patient_info_df["正式编号"] = patient_info_df["正式编号"].astype(str).str.strip()
        if "患者大类型" in patient_info_df.columns:
            patient_info_df["患者大类型"] = patient_info_df["患者大类型"].apply(normalize_patient_type)

        if {"正式编号", "患者大类型"}.issubset(patient_info_df.columns):
            patient_summary = patient_summary.merge(
                patient_info_df[["正式编号", "患者大类型"]].drop_duplicates(),
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

    return patient_summary, cell_df


def build_patient_celltype_stats(cell_df: pd.DataFrame):
    df = cell_df.copy()
    required_cols = ["patient_id", "cell_type", "mapped_label", "pred_label", "image"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    df["is_correct"] = (df["mapped_label"] == df["pred_label"]).astype(int)
    df["is_wrong"] = 1 - df["is_correct"]

    detail = (
        df.groupby(["patient_id", "cell_type"], dropna=False)
        .agg(
            n_cells=("image", "count"),
            n_correct=("is_correct", "sum"),
            n_wrong=("is_wrong", "sum"),
            true_binary_label=("mapped_label", "first"),
            pred_positive_ratio=("pred_label", "mean"),
        )
        .reset_index()
    )

    detail["correct_ratio"] = detail["n_correct"] / detail["n_cells"]
    detail["wrong_ratio"] = detail["n_wrong"] / detail["n_cells"]
    detail = detail.sort_values(["patient_id", "cell_type"]).reset_index(drop=True)

    wide_count = detail.pivot(index="patient_id", columns="cell_type", values="n_cells").add_prefix("count_")
    wide_correct = detail.pivot(index="patient_id", columns="cell_type", values="n_correct").add_prefix("n_correct_")
    wide_wrong = detail.pivot(index="patient_id", columns="cell_type", values="n_wrong").add_prefix("n_wrong_")
    wide_correct_ratio = detail.pivot(index="patient_id", columns="cell_type", values="correct_ratio").add_prefix("correct_ratio_")
    wide_wrong_ratio = detail.pivot(index="patient_id", columns="cell_type", values="wrong_ratio").add_prefix("wrong_ratio_")

    wide = pd.concat(
        [wide_count, wide_correct, wide_wrong, wide_correct_ratio, wide_wrong_ratio],
        axis=1,
    ).reset_index()
    wide = wide.sort_values("patient_id").reset_index(drop=True)

    return detail, wide


def plot_patient_ratios(plot_df: pd.DataFrame, save_path: Path, title: str):
    target_types = ["AML", "HC"]
    if "type" in plot_df.columns and plot_df["type"].isin(target_types).any():
        plot_df = plot_df[plot_df["type"].isin(target_types)].copy()
    else:
        plot_df = plot_df.copy()
        if "type" not in plot_df.columns:
            plot_df["type"] = np.nan

    best_thresh = 0
    if len(plot_df) > 0 and "AML" in plot_df["type"].unique() and "HC" in plot_df["type"].unique():
        try:
            y_true = (plot_df["type"] == "AML").astype(int)
            y_score = plot_df["predicted_ratio"]
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            if len(thresholds) > 0:
                j_stat = tpr - fpr
                best_thresh = thresholds[np.argmax(j_stat)]
        except Exception as e:
            print(f"⚠️ 阈值计算失败: {e}")

    plot_df = plot_df.sort_values("predicted_ratio", ascending=True).reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(18, len(plot_df) * 0.7), 9))
    ax = plt.gca()

    rects1 = ax.bar(
        x - width / 2,
        plot_df["actual_ratio"],
        width,
        color="lightcoral",
        edgecolor="red",
        label="实际原始细胞比例",
    )
    rects2 = ax.bar(
        x + width / 2,
        plot_df["predicted_ratio"],
        width,
        color="lightblue",
        edgecolor="blue",
        label="预测原始细胞比例",
    )

    def add_value_labels(rects, offset=0.02):
        for rect in rects:
            height = rect.get_height()
            if pd.notna(height) and height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + offset,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_value_labels(rects1)
    add_value_labels(rects2)

    xtick_labels = [
        f"{str(pid)[:20]}\n{t if pd.notna(t) else 'NA'}"
        for pid, t in zip(plot_df["patient_id"], plot_df["type"])
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylabel("原始细胞比例", fontsize=12)
    ax.set_xlabel("患者 (按预测比例排序)", fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(
        handles=[
            Line2D([0], [0], color="lightcoral", lw=4, label="实际原始细胞比例"),
            Line2D([0], [0], color="lightblue", lw=4, label="预测原始细胞比例"),
        ],
        bbox_to_anchor=(1.0, 1),
        loc="upper right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for lbl, t in zip(ax.get_xticklabels(), plot_df["type"]):
        if t == "AML":
            lbl.set_color("red")
        elif t == "HC":
            lbl.set_color("blue")

    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ 患者比例图已保存: {save_path}")
    if best_thresh > 0:
        print(f"📊 AML/HC 最佳区分阈值: {best_thresh:.4f}")

    return plot_df, best_thresh


def run_patient_analysis(
    merged_df: pd.DataFrame,
    out_dir: Path,
    patient_info_xlsx: Optional[Path],
):
    if not {"image", "pred_label"}.issubset(merged_df.columns):
        print("⚠️ 缺少 image/pred_label，跳过患者级分析")
        return None

    patient_info_df = None
    if patient_info_xlsx is not None:
        if Path(patient_info_xlsx).exists():
            patient_info_df = pd.read_excel(patient_info_xlsx)
            print(f"📖 已读取患者信息: {patient_info_xlsx}")
        else:
            print(f"⚠️ 患者信息文件不存在，继续做无类型患者分析: {patient_info_xlsx}")

    patient_summary, cell_df_with_info = build_patient_summary(merged_df, patient_info_df)
    patient_celltype_detail, patient_celltype_wide = build_patient_celltype_stats(cell_df_with_info)

    patient_type_count_summary = (
        patient_celltype_detail.groupby("patient_id")
        .agg(
            total_cells=("n_cells", "sum"),
            total_types=("cell_type", "nunique"),
            total_correct=("n_correct", "sum"),
            total_wrong=("n_wrong", "sum"),
        )
        .reset_index()
    )
    patient_type_count_summary["overall_correct_ratio"] = (
        patient_type_count_summary["total_correct"] / patient_type_count_summary["total_cells"]
    )
    patient_type_count_summary["overall_wrong_ratio"] = (
        patient_type_count_summary["total_wrong"] / patient_type_count_summary["total_cells"]
    )

    patient_excel = out_dir / "patient_analysis.xlsx"
    with pd.ExcelWriter(patient_excel, engine="openpyxl") as writer:
        patient_summary.to_excel(writer, sheet_name="patient_summary", index=False)
        patient_celltype_detail.to_excel(writer, sheet_name="patient_celltype_detail", index=False)
        patient_celltype_wide.to_excel(writer, sheet_name="patient_celltype_wide", index=False)
        patient_type_count_summary.to_excel(writer, sheet_name="patient_type_count_summary", index=False)

    patient_summary.to_csv(out_dir / "patient_summary.csv", index=False)
    patient_celltype_detail.to_csv(out_dir / "patient_celltype_detail.csv", index=False)
    patient_celltype_wide.to_csv(out_dir / "patient_celltype_wide.csv", index=False)
    patient_type_count_summary.to_csv(out_dir / "patient_type_count_summary.csv", index=False)

    plot_patient_ratios(
        plot_df=patient_summary,
        save_path=out_dir / "patient_ratio_from_cell_results.png",
        title=f"{out_dir.name} 各患者原始细胞比例对比 (实际 vs 预测)",
    )

    print(f"✅ 患者级统计已保存到: {patient_excel}")
    return patient_summary


# =========================
# 5-fold vs 非5-fold 对比
# =========================
def resolve_single_cell_csv(single_root: Path, split_name: str) -> Path:
    """
    支持两种传参方式：
    1) --single-root 指向总目录：<single_root>/<split_name>/val_results.csv
    2) --single-root 直接指向某个 split 目录：<single_root>/val_results.csv
    """
    candidate_1 = single_root / split_name / "val_results.csv"
    candidate_2 = single_root / "val_results.csv"

    if candidate_1.exists():
        return candidate_1
    if candidate_2.exists():
        return candidate_2
    raise FileNotFoundError(
        f"找不到非5-fold细胞级结果，已尝试:\n- {candidate_1}\n- {candidate_2}"
    )


def build_single_model_patient_summary(
    cell_df: pd.DataFrame,
    patient_info_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    patient_summary, _ = build_patient_summary(cell_df, patient_info_df)
    return patient_summary.rename(
        columns={
            "n_cells": "single_n_cells",
            "n_smears": "single_n_smears",
            "actual_ratio": "single_actual_ratio",
            "predicted_ratio": "single_predicted_ratio",
            "mean_prob_class_1": "single_mean_prob_class_1",
            "accuracy": "single_accuracy",
        }
    )


def build_compare_table(
    ensemble_patient_df: pd.DataFrame,
    single_patient_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ensemble_df = ensemble_patient_df.copy().rename(
        columns={
            "n_cells": "ensemble_n_cells",
            "n_smears": "ensemble_n_smears",
            "actual_ratio": "ensemble_actual_ratio",
            "predicted_ratio": "ensemble_predicted_ratio",
            "mean_prob_class_1": "ensemble_mean_prob_class_1",
            "accuracy": "ensemble_accuracy",
        }
    )

    merged = ensemble_df.merge(single_patient_df, on="patient_id", how="outer", suffixes=("", "_single"))

    if "type" in merged.columns and "type_single" in merged.columns:
        merged["type"] = merged["type"].fillna(merged["type_single"])
        merged.drop(columns=["type_single"], inplace=True)
    elif "type_single" in merged.columns and "type" not in merged.columns:
        merged.rename(columns={"type_single": "type"}, inplace=True)

    merged["actual_ratio"] = merged.get("ensemble_actual_ratio")
    if "single_actual_ratio" in merged.columns:
        merged["actual_ratio"] = merged["actual_ratio"].fillna(merged["single_actual_ratio"])

    merged["split_name"] = split_name
    merged["abs_error_5fold"] = (merged["ensemble_predicted_ratio"] - merged["actual_ratio"]).abs()
    merged["abs_error_single"] = (merged["single_predicted_ratio"] - merged["actual_ratio"]).abs()
    merged["delta_abs_error"] = merged["abs_error_single"] - merged["abs_error_5fold"]

    return merged.sort_values("patient_id").reset_index(drop=True)


def plot_three_ratio_bars(plot_df: pd.DataFrame, save_path: Path, title: str):
    plot_df = plot_df.copy()

    if "type" in plot_df.columns and plot_df["type"].isin(["AML", "HC"]).any():
        plot_df = plot_df[plot_df["type"].isin(["AML", "HC"])].copy()
    elif "type" not in plot_df.columns:
        plot_df["type"] = np.nan

    plot_df["sort_key"] = plot_df["ensemble_predicted_ratio"]
    plot_df["sort_key"] = plot_df["sort_key"].fillna(plot_df["single_predicted_ratio"])
    plot_df = plot_df.sort_values("sort_key", ascending=True).reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.26

    plt.figure(figsize=(max(18, len(plot_df) * 0.75), 9))
    ax = plt.gca()

    rects1 = ax.bar(
        x - width,
        plot_df["actual_ratio"],
        width,
        color="lightcoral",
        edgecolor="red",
        label="实际原始细胞比例",
    )
    rects2 = ax.bar(
        x,
        plot_df["ensemble_predicted_ratio"],
        width,
        color="lightblue",
        edgecolor="blue",
        label="5-fold预测原始细胞比例",
    )
    rects3 = ax.bar(
        x + width,
        plot_df["single_predicted_ratio"],
        width,
        color="lightgreen",
        edgecolor="green",
        label="非5-fold预测原始细胞比例",
    )

    def add_value_labels(rects, offset=0.015):
        for rect in rects:
            height = rect.get_height()
            if pd.notna(height) and height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + offset,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    add_value_labels(rects1)
    add_value_labels(rects2)
    add_value_labels(rects3)

    if "type" in plot_df.columns:
        xtick_labels = [
            f"{str(pid)[:20]}\n{t if pd.notna(t) else 'NA'}"
            for pid, t in zip(plot_df["patient_id"], plot_df["type"])
        ]
    else:
        xtick_labels = [str(pid)[:20] for pid in plot_df["patient_id"]]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylabel("原始细胞比例", fontsize=12)
    ax.set_xlabel("患者 (按5-fold预测比例排序)", fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(
        handles=[
            Line2D([0], [0], color="lightcoral", lw=4, label="实际原始细胞比例"),
            Line2D([0], [0], color="lightblue", lw=4, label="5-fold预测原始细胞比例"),
            Line2D([0], [0], color="lightgreen", lw=4, label="非5-fold预测原始细胞比例"),
        ],
        bbox_to_anchor=(1.0, 1),
        loc="upper right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if "type" in plot_df.columns:
        for lbl, t in zip(ax.get_xticklabels(), plot_df["type"]):
            if t == "AML":
                lbl.set_color("red")
            elif t == "HC":
                lbl.set_color("blue")

    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ 三组对比图已保存: {save_path}")


def run_compare_analysis(
    split_name: str,
    ensemble_patient_df: pd.DataFrame,
    single_root: Path,
    out_dir: Path,
    patient_info_df: Optional[pd.DataFrame],
):
    single_cell_csv = resolve_single_cell_csv(single_root, split_name)
    print(f"📖 [{split_name}] 读取非5-fold结果: {single_cell_csv}")
    single_cell_df = pd.read_csv(single_cell_csv)

    required_cols_csv = ["image", "pred_label"]
    for col in required_cols_csv:
        if col not in single_cell_df.columns:
            raise ValueError(f"[{split_name}] 非5-fold结果缺少必要列: {col}")

    single_patient_df = build_single_model_patient_summary(single_cell_df, patient_info_df)
    compare_df = build_compare_table(ensemble_patient_df, single_patient_df, split_name=split_name)

    compare_excel = out_dir / "patient_ratio_compare_3bars.xlsx"
    with pd.ExcelWriter(compare_excel, engine="openpyxl") as writer:
        compare_df.to_excel(writer, sheet_name="compare_summary", index=False)
        ensemble_patient_df.to_excel(writer, sheet_name="ensemble_patient", index=False)
        single_patient_df.to_excel(writer, sheet_name="single_patient", index=False)

    compare_df.to_csv(out_dir / "patient_ratio_compare_3bars.csv", index=False)

    plot_three_ratio_bars(
        compare_df,
        out_dir / "patient_ratio_compare_3bars.png",
        title=f"{split_name}：各患者原始细胞比例对比（真实 vs 5-fold预测 vs 非5-fold预测）",
    )

    print(f"✅ [{split_name}] 对比结果已保存: {compare_excel}")
    return compare_df


def save_compare_batch_summary(compare_dfs: List[pd.DataFrame], out_root: Path):
    if len(compare_dfs) == 0:
        return

    all_df = pd.concat(compare_dfs, axis=0, ignore_index=True)
    summary_excel = out_root / "all_splits_compare_summary.xlsx"

    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="all_compare_summary", index=False)

        for split_name in all_df["split_name"].dropna().unique().tolist():
            sub = all_df[all_df["split_name"] == split_name].copy()
            if len(sub) > 0:
                safe_sheet = str(split_name)[:31]
                sub.to_excel(writer, sheet_name=safe_sheet, index=False)

        metric_rows = []
        for split_name in all_df["split_name"].dropna().unique().tolist():
            sub = all_df[all_df["split_name"] == split_name].copy()
            if len(sub) == 0:
                continue

            row = {
                "split_name": split_name,
                "n_patients": len(sub),
                "mean_abs_error_5fold": sub["abs_error_5fold"].mean(),
                "mean_abs_error_single": sub["abs_error_single"].mean(),
                "median_abs_error_5fold": sub["abs_error_5fold"].median(),
                "median_abs_error_single": sub["abs_error_single"].median(),
                "mean_delta_abs_error(single-5fold)": sub["delta_abs_error"].mean(),
                "n_5fold_better": int((sub["delta_abs_error"] > 0).sum()),
                "n_single_better": int((sub["delta_abs_error"] < 0).sum()),
                "n_equal": int((sub["delta_abs_error"] == 0).sum()),
            }
            metric_rows.append(row)

        metric_df = pd.DataFrame(metric_rows)
        metric_df.to_excel(writer, sheet_name="error_summary", index=False)

    all_df.to_csv(out_root / "all_splits_compare_summary.csv", index=False)
    print(f"✅ 全部 split 的5-fold vs 非5-fold汇总已保存: {summary_excel}")


# =========================
# 单个 split 的处理
# =========================
def merge_one_split(
    root: Path,
    split_name: str,
    out_root: Path,
    binary_threshold: float,
    patient_info_xlsx: Optional[Path] = None,
    single_root: Optional[Path] = None,
    patient_info_df_cache: Optional[pd.DataFrame] = None,
):
    fold_csvs = discover_fold_csvs(root, split_name)
    if len(fold_csvs) == 0:
        print(f"⚠️ [{split_name}] 未找到 fold 结果，跳过")
        return None, None

    print(f"\n{'=' * 100}")
    print(f"🚀 开始处理 split: {split_name}")
    for p in fold_csvs:
        print(f"   - {p}")

    dfs = [pd.read_csv(p) for p in fold_csvs]
    has_probs = all(any(c.startswith("prob_class_") for c in df.columns) for df in dfs)

    if has_probs:
        merged = merge_with_probs(dfs, binary_threshold=binary_threshold)
        print(f"✅ [{split_name}] 使用概率融合（平均概率）")
    else:
        merged = merge_with_vote(dfs)
        print(f"✅ [{split_name}] 无概率列，使用多数投票融合")

    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "val_results.csv"
    merged.to_csv(out_csv, index=False)
    print(f"📄 [{split_name}] 细胞级融合结果已保存: {out_csv}")

    if "true_label" in merged.columns:
        prob_cols = [c for c in merged.columns if c.startswith("prob_class_")]
        num_classes = len(prob_cols) if len(prob_cols) > 0 else int(merged["pred_label"].max()) + 1
        calculate_and_save_metrics(
            df=merged,
            output_dir=str(out_dir),
            split="val",
            timestamp="ensemble",
            num_classes=num_classes,
        )
        print(f"📊 [{split_name}] metrics.csv 已生成")
    else:
        print(f"⚠️ [{split_name}] 无 true_label，跳过 calculate_and_save_metrics")

    patient_summary = run_patient_analysis(
        merged_df=merged,
        out_dir=out_dir,
        patient_info_xlsx=patient_info_xlsx,
    )

    compare_df = None
    if single_root is not None and patient_summary is not None:
        try:
            compare_df = run_compare_analysis(
                split_name=split_name,
                ensemble_patient_df=patient_summary,
                single_root=single_root,
                out_dir=out_dir,
                patient_info_df=patient_info_df_cache,
            )
        except Exception as e:
            print(f"❌ [{split_name}] 生成5-fold vs 非5-fold对比失败: {e}")

    return patient_summary, compare_df


def parse_args():
    p = argparse.ArgumentParser(description="融合 5-fold 结果，追加患者级分析，并可与非5-fold结果对比")
    p.add_argument(
        "--root",
        default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold_260403",
        help="包含 fold*/eval 的 5-fold 根目录",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="输出目录，默认 <root>/ensemble_eval",
    )
    p.add_argument(
        "--single-root",
        default="/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/260323_gt2yolo_576_0.65_2class_onlineAug",
        help="非5-fold结果目录。支持 <single_root>/<split>/val_results.csv，或直接 <single_root>/val_results.csv",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=["test_BJH", "test_FXH_noALL", "test_TJMU"],
        help="待处理的 split 名称",
    )
    p.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="二分类 prob_class_1 阈值",
    )
    p.add_argument(
        "--patient-info-xlsx",
        default="/root/autodl-tmp/data/patient_data_260323.xlsx",
        help="患者信息表；若不存在则仍会输出无类型患者统计",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_chinese_font()

    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else (root / "ensemble_eval")
    patient_info_xlsx = Path(args.patient_info_xlsx) if args.patient_info_xlsx else None
    single_root = Path(args.single_root) if args.single_root else None

    patient_info_df_cache = None
    if patient_info_xlsx is not None and patient_info_xlsx.exists():
        patient_info_df_cache = pd.read_excel(patient_info_xlsx)

    all_compare_dfs: List[pd.DataFrame] = []

    for split_name in args.splits:
        _, compare_df = merge_one_split(
            root=root,
            split_name=split_name,
            out_root=out_root,
            binary_threshold=args.binary_threshold,
            patient_info_xlsx=patient_info_xlsx,
            single_root=single_root,
            patient_info_df_cache=patient_info_df_cache,
        )
        if compare_df is not None:
            all_compare_dfs.append(compare_df)

    if len(all_compare_dfs) > 0:
        save_compare_batch_summary(all_compare_dfs, out_root)


if __name__ == "__main__":
    main()
