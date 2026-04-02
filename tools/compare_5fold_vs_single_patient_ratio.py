import os
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# =========================
# 全局配置
# =========================

font_path = "/root/autodl-tmp/projects/myq/SingleCellProject/tools/MSYH.TTC"
FONT_NAME = "MSYH.TTC"
POSITIVE_CLASS = 1

cell_dict_big = {
    "V": 0, "0": 0,
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
}


# =========================
# 基础工具
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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
            font_name = font_prop.get_name()
            plt.rcParams["font.family"] = font_name
            print(f"✅ 已加载中文字体: {font_name}")
        except Exception as e:
            print(f"⚠️ 字体加载失败，使用回退字体: {e}")
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    else:
        print("⚠️ 未找到中文字体文件，使用回退字体")
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]

    plt.rcParams["axes.unicode_minus"] = False


def normalize_patient_type(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()
    if x in ["HC", "HD", "NORMAL", "HEALTHY"]:
        return "HC"
    elif x in ["AML"]:
        return "AML"
    else:
        return x


# =========================
# 参考上一问 patient_analysis 的解析逻辑
# =========================

def parse_image_info(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    例:
    PKUPH-106-10_000_P2 -> patient_id=PKUPH-106, smear_id=10
    FXH-1_000_P2 -> patient_id=FXH-1, smear_id=None
    """
    image_name = str(image_name)
    stem = Path(image_name).stem
    prefix = stem.split("_")[0]
    parts = prefix.split("-")

    if len(parts) >= 3:
        patient_id = f"{parts[0]}-{parts[1]}"
        smear_id = parts[2]
        return patient_id, smear_id
    elif len(parts) >= 2:
        patient_id = f"{parts[0]}-{parts[1]}"
        return patient_id, None

    return None, None


def parse_cell_type(image_name: str):
    """
    例:
    PKUPH-106-10_000_P2 -> P2
    """
    if pd.isna(image_name):
        return None

    image_name = str(image_name).strip()
    stem = Path(image_name).stem
    if "_" not in stem:
        return None

    return stem.split("_")[-1].strip().upper()


def map_cell_type_to_binary(cell_type: str):
    if pd.isna(cell_type) or cell_type is None:
        return 0

    v = cell_dict_big.get(str(cell_type).strip().upper(), 0)
    return 1 if v == 1 else 0


# =========================
# 单模型：细胞级 -> 患者级
# =========================

def build_single_model_patient_summary(cell_df: pd.DataFrame, patient_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    参考你给的 patient_analysis.py 思路：
    - actual_ratio 从 image -> cell_type -> mapped_label 得到
    - predicted_ratio 从 pred_label 得到
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

    patient_summary = (
        cell_df.groupby("patient_id")
        .agg(
            single_n_cells=("image", "count"),
            single_actual_ratio=("actual_positive", "mean"),
            single_predicted_ratio=("pred_positive", "mean"),
        )
        .reset_index()
    )

    patient_info_df = patient_info_df.copy()
    patient_info_df["正式编号"] = patient_info_df["正式编号"].astype(str).str.strip()
    patient_info_df["患者大类型"] = patient_info_df["患者大类型"].apply(normalize_patient_type)

    patient_summary = patient_summary.merge(
        patient_info_df[["正式编号", "患者大类型"]].drop_duplicates(),
        left_on="patient_id",
        right_on="正式编号",
        how="left"
    )
    patient_summary.rename(columns={"患者大类型": "type"}, inplace=True)
    patient_summary.drop(columns=["正式编号"], inplace=True)

    return patient_summary


# =========================
# 读取 ensemble 结果
# =========================

def load_ensemble_patient_summary(ensemble_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ensemble_csv).copy()

    required_cols = ["patient_id", "actual_ratio", "predicted_ratio"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"ensemble csv 缺少必要列: {col}")

    rename_map = {
        "actual_ratio": "ensemble_actual_ratio",
        "predicted_ratio": "ensemble_predicted_ratio",
    }

    if "type" in df.columns:
        keep_cols = ["patient_id", "actual_ratio", "predicted_ratio", "type"]
    else:
        keep_cols = ["patient_id", "actual_ratio", "predicted_ratio"]

    df = df[keep_cols].rename(columns=rename_map)

    return df


# =========================
# 合并对比表
# =========================

def build_compare_table(
    ensemble_df: pd.DataFrame,
    single_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    merged = ensemble_df.merge(single_df, on="patient_id", how="outer", suffixes=("", "_single"))

    # type 优先用 ensemble，再用 single
    if "type" in merged.columns and "type_single" in merged.columns:
        merged["type"] = merged["type"].fillna(merged["type_single"])
        merged.drop(columns=["type_single"], inplace=True)
    elif "type_single" in merged.columns and "type" not in merged.columns:
        merged.rename(columns={"type_single": "type"}, inplace=True)

    # 真实比例优先取 ensemble 的；若缺失则用 single 的
    merged["actual_ratio"] = merged["ensemble_actual_ratio"]
    if "single_actual_ratio" in merged.columns:
        merged["actual_ratio"] = merged["actual_ratio"].fillna(merged["single_actual_ratio"])

    merged["split_name"] = split_name

    # 误差
    merged["abs_error_5fold"] = (merged["ensemble_predicted_ratio"] - merged["actual_ratio"]).abs()
    merged["abs_error_single"] = (merged["single_predicted_ratio"] - merged["actual_ratio"]).abs()

    return merged


# =========================
# 作图
# =========================

def plot_three_ratio_bars(plot_df: pd.DataFrame, save_path: Path, title: str):
    """
    三根柱子：
    1) 真实比例
    2) 5-fold预测比例
    3) 单模型预测比例
    """
    plot_df = plot_df.copy()

    if "type" in plot_df.columns and plot_df["type"].isin(["AML", "HC"]).any():
        plot_df = plot_df[plot_df["type"].isin(["AML", "HC"])].copy()

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
        label="实际原始细胞比例"
    )

    rects2 = ax.bar(
        x,
        plot_df["ensemble_predicted_ratio"],
        width,
        color="lightblue",
        edgecolor="blue",
        label="5-fold预测原始细胞比例"
    )

    rects3 = ax.bar(
        x + width,
        plot_df["single_predicted_ratio"],
        width,
        color="lightgreen",
        edgecolor="green",
        label="单个训练预测原始细胞比例"
    )

    def add_value_labels(rects, offset=0.015):
        for rect in rects:
            height = rect.get_height()
            if pd.notna(height) and height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.,
                    height + offset,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

    add_value_labels(rects1)
    add_value_labels(rects2)
    add_value_labels(rects3)

    legend_elements = [
        Line2D([0], [0], color="lightcoral", lw=4, label="实际原始细胞比例"),
        Line2D([0], [0], color="lightblue", lw=4, label="5-fold预测原始细胞比例"),
        Line2D([0], [0], color="lightgreen", lw=4, label="单个训练预测原始细胞比例"),
    ]

    ax.set_ylabel("原始细胞比例", fontsize=12)
    ax.set_xlabel("患者 (按5-fold预测比例排序)", fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14, pad=20)

    if "type" in plot_df.columns:
        xtick_labels = [
            f"{str(pid)[:20]}\n{t if pd.notna(t) else 'NA'}"
            for pid, t in zip(plot_df["patient_id"], plot_df["type"])
        ]
    else:
        xtick_labels = [str(pid)[:20] for pid in plot_df["patient_id"]]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    if "type" in plot_df.columns:
        for lbl, t in zip(ax.get_xticklabels(), plot_df["type"]):
            if t == "AML":
                lbl.set_color("red")
            elif t == "HC":
                lbl.set_color("blue")

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ 三组对比图已保存: {save_path}")


# =========================
# 单个 split 处理
# =========================

def run_one_split(
    split_name: str,
    ensemble_root: Path,
    single_root: Path,
    patient_info_xlsx: Path,
):
    print("\n" + "=" * 100)
    print(f"🚀 开始处理 split: {split_name}")
    print("=" * 100)

    ensemble_csv = ensemble_root / split_name / "patient_level_ensemble.csv"
    single_cell_csv = single_root / split_name / "val_results.csv"
    out_png = ensemble_root / split_name / "patient_ratio_compare_3bars.png"
    out_excel = ensemble_root / split_name / "patient_ratio_compare_3bars.xlsx"

    print(f"ensemble_csv    : {ensemble_csv}")
    print(f"single_cell_csv : {single_cell_csv}")
    print(f"out_png         : {out_png}")
    print(f"out_excel       : {out_excel}")

    if not ensemble_csv.exists():
        raise FileNotFoundError(f"找不到 ensemble csv: {ensemble_csv}")
    if not single_cell_csv.exists():
        raise FileNotFoundError(f"找不到 single model 细胞级 csv: {single_cell_csv}")
    if not patient_info_xlsx.exists():
        raise FileNotFoundError(f"找不到患者信息表: {patient_info_xlsx}")

    ensure_parent_dir(out_png)
    ensure_parent_dir(out_excel)

    ensemble_df = load_ensemble_patient_summary(ensemble_csv)
    single_cell_df = pd.read_csv(single_cell_csv)
    patient_info_df = pd.read_excel(patient_info_xlsx)

    required_cols_csv = ["image", "pred_label"]
    for col in required_cols_csv:
        if col not in single_cell_df.columns:
            raise ValueError(f"{split_name} 的 single cell csv 缺少必要列: {col}")

    required_cols_xlsx = ["正式编号", "患者大类型"]
    for col in required_cols_xlsx:
        if col not in patient_info_df.columns:
            raise ValueError(f"患者信息表缺少必要列: {col}")

    single_patient_df = build_single_model_patient_summary(single_cell_df, patient_info_df)
    compare_df = build_compare_table(ensemble_df, single_patient_df, split_name=split_name)

    # 保存 excel
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        compare_df.to_excel(writer, sheet_name="compare_summary", index=False)
        ensemble_df.to_excel(writer, sheet_name="ensemble_patient", index=False)
        single_patient_df.to_excel(writer, sheet_name="single_patient", index=False)

    print(f"✅ 对比表已保存: {out_excel}")

    plot_three_ratio_bars(
        compare_df,
        out_png,
        title=f"{split_name}：各患者原始细胞比例对比（真实 vs 5-fold预测 vs 单个训练预测）"
    )

    print("\n===== 对比结果预览 =====")
    print(compare_df.head())

    return compare_df


# =========================
# 主函数
# =========================

def main():
    set_chinese_font()

    # 直接使用之前代码中出现过的路径
    ensemble_root = Path("/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold/ensemble_eval_patient_level")
    single_root = Path("/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/260323_gt2yolo_576_0.65_2class_onlineAug")
    patient_info_xlsx = Path("/root/autodl-tmp/data/样本信息整理260323.xlsx")

    splits = ["test_BJH", "test_FXH_noALL", "test_TJMU"]

    all_compare_dfs: List[pd.DataFrame] = []
    failed = []

    print(f"ensemble_root   = {ensemble_root}")
    print(f"single_root     = {single_root}")
    print(f"patient_info_xlsx = {patient_info_xlsx}")
    print(f"splits          = {splits}")

    for split_name in splits:
        try:
            compare_df = run_one_split(
                split_name=split_name,
                ensemble_root=ensemble_root,
                single_root=single_root,
                patient_info_xlsx=patient_info_xlsx,
            )
            all_compare_dfs.append(compare_df)
        except Exception as e:
            print(f"❌ {split_name} 处理失败: {e}")
            failed.append({
                "split_name": split_name,
                "error": str(e)
            })

    # 保存汇总总表
    if len(all_compare_dfs) > 0:
        all_df = pd.concat(all_compare_dfs, axis=0, ignore_index=True)

        summary_excel = ensemble_root / "all_splits_compare_summary.xlsx"
        with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="all_compare_summary", index=False)

            # 每个split单独一张sheet
            for split_name in splits:
                sub = all_df[all_df["split_name"] == split_name].copy()
                if len(sub) > 0:
                    safe_sheet = split_name[:31]
                    sub.to_excel(writer, sheet_name=safe_sheet, index=False)

            # 误差统计
            metric_rows = []
            for split_name in splits:
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
                }
                metric_rows.append(row)

            metric_df = pd.DataFrame(metric_rows)
            metric_df.to_excel(writer, sheet_name="error_summary", index=False)

        print(f"\n✅ 全部split汇总表已保存: {summary_excel}")

    if len(failed) > 0:
        failed_df = pd.DataFrame(failed)
        failed_csv = ensemble_root / "batch_compare_failed.csv"
        failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"⚠️ 部分任务失败，已保存: {failed_csv}")

    print("\n🎉 批处理完成")


if __name__ == "__main__":
    main()