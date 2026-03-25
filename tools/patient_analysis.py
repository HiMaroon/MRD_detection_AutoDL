import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D

# =========================
# 全局配置
# =========================

# 字体文件
font_path = "/root/autodl-tmp/projects/myq/SingleCellProject/tools/MSYH.TTC"
FONT_NAME = "MSYH.TTC"

# 预测为哪一类算“原始细胞/AML类”
POSITIVE_CLASS = 1

# 细胞类型映射
# 规则：
# - 只有 value == 1 的算正类(1)
# - value == 0 / 2 的都算负类(0)
# - 不在字典中的也统一算 0
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
# 工具函数
# =========================

def ensure_parent_dir(file_path):
    """自动创建文件的父目录"""
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"📁 已创建输出目录: {parent_dir}")


def set_chinese_font():
    """尝试加载中文字体，解决乱码问题"""
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
        print(f"✅ 找到字体文件: {selected_font}")
        try:
            fm.fontManager.addfont(selected_font)
            font_prop = fm.FontProperties(fname=selected_font)
            font_name = font_prop.get_name()
            plt.rcParams["font.family"] = font_name
            print(f"   已注册并设置为默认字体: {font_name}")
        except Exception as e:
            print(f"⚠️ 字体注册异常: {e}")
    else:
        print(f"⚠️ 未找到 {FONT_NAME}，尝试使用系统回退字体")
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]

    plt.rcParams["axes.unicode_minus"] = False


def parse_image_info(image_name: str):
    """
    例:
    PKUPH-106-10_000_P2

    解析结果:
    patient_id = PKUPH-106
    smear_id   = 10
    """
    image_name = str(image_name)
    prefix = image_name.split("_")[0]
    parts = prefix.split("-")

    if len(parts) < 3:
        return None, None

    patient_id = f"{parts[0]}-{parts[1]}"
    smear_id = parts[2]
    return patient_id, smear_id


def parse_cell_type(image_name: str):
    """
    从图像名最后一个下划线后提取细胞类型

    例:
    PKUPH-106-10_000_P2 -> P2
    """
    if pd.isna(image_name):
        return None

    image_name = str(image_name).strip()
    if "_" not in image_name:
        return None

    return image_name.split("_")[-1].strip().upper()


def map_cell_type_to_binary(cell_type: str):
    """
    将 cell_type 映射为二分类：
    - value == 1 -> 1
    - value == 0 / 2 / 不在字典中 -> 0
    """
    if pd.isna(cell_type) or cell_type is None:
        return 0

    v = cell_dict_big.get(str(cell_type).strip().upper(), 0)
    return 1 if v == 1 else 0


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
# 核心统计函数
# =========================

def build_patient_summary(cell_df, patient_info_df):
    """
    从细胞级结果构建患者级汇总表

    注意：
    这里的 actual_ratio 不再依赖 true_label，
    而是依赖 image 名解析得到的 cell_type -> mapped_label
    """
    cell_df = cell_df.copy()

    # 解析 patient_id / smear_id
    parsed = cell_df["image"].apply(parse_image_info)
    cell_df["patient_id"] = parsed.apply(lambda x: x[0])
    cell_df["smear_id"] = parsed.apply(lambda x: x[1])

    # 解析 cell_type，并映射成二分类标签
    cell_df["cell_type"] = cell_df["image"].apply(parse_cell_type)
    cell_df["mapped_label"] = cell_df["cell_type"].apply(map_cell_type_to_binary)

    bad_rows = cell_df["patient_id"].isna().sum()
    if bad_rows > 0:
        print(f"⚠️ 有 {bad_rows} 条记录无法解析 patient_id，已忽略")
        cell_df = cell_df[cell_df["patient_id"].notna()].copy()

    # 基于 cell_type 映射标签做统计
    cell_df["actual_positive"] = (cell_df["mapped_label"] == POSITIVE_CLASS).astype(int)
    cell_df["pred_positive"] = (cell_df["pred_label"] == POSITIVE_CLASS).astype(int)
    cell_df["is_correct"] = (cell_df["mapped_label"] == cell_df["pred_label"]).astype(int)

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

    # 合并患者类型
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

    return patient_summary, cell_df


def build_patient_celltype_stats(cell_df):
    """
    构建每个患者、每种细胞类型的统计表：
    - 细胞个数
    - 正确个数 / 错误个数
    - 正确比例 / 错误比例

    返回：
    1) detail: 长表
    2) wide:   宽表
    """
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

    # 宽表
    wide_count = detail.pivot(index="patient_id", columns="cell_type", values="n_cells")
    wide_count = wide_count.add_prefix("count_")

    wide_correct = detail.pivot(index="patient_id", columns="cell_type", values="n_correct")
    wide_correct = wide_correct.add_prefix("n_correct_")

    wide_wrong = detail.pivot(index="patient_id", columns="cell_type", values="n_wrong")
    wide_wrong = wide_wrong.add_prefix("n_wrong_")

    wide_correct_ratio = detail.pivot(index="patient_id", columns="cell_type", values="correct_ratio")
    wide_correct_ratio = wide_correct_ratio.add_prefix("correct_ratio_")

    wide_wrong_ratio = detail.pivot(index="patient_id", columns="cell_type", values="wrong_ratio")
    wide_wrong_ratio = wide_wrong_ratio.add_prefix("wrong_ratio_")

    wide = pd.concat(
        [wide_count, wide_correct, wide_wrong, wide_correct_ratio, wide_wrong_ratio],
        axis=1
    ).reset_index()

    wide = wide.sort_values("patient_id").reset_index(drop=True)

    return detail, wide


# =========================
# 绘图
# =========================

def plot_patient_ratios(plot_df, save_path, title="各患者原始细胞比例对比 (实际 vs 预测)"):
    """
    仿照原脚本风格绘图，但把 AML/HC 并入 x 轴标签第二行，避免重叠
    """
    target_types = ["AML", "HC"]
    if plot_df["type"].isin(target_types).any():
        plot_df = plot_df[plot_df["type"].isin(target_types)].copy()
    else:
        plot_df = plot_df.copy()

    best_thresh = 0
    if "AML" in plot_df["type"].unique() and "HC" in plot_df["type"].unique():
        try:
            y_true = (plot_df["type"] == "AML").astype(int)
            y_score = plot_df["predicted_ratio"]
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            if len(thresholds) > 0:
                J = tpr - fpr
                best_thresh = thresholds[np.argmax(J)]
        except Exception as e:
            print(f"⚠️ 阈值计算失败: {e}")

    plot_df = plot_df.sort_values("predicted_ratio", ascending=True).reset_index(drop=True)
    plot_df["color"] = plot_df["type"].map({"AML": "red", "HC": "blue"}).fillna("black")

    plt.figure(figsize=(max(18, len(plot_df) * 0.7), 9))
    ax = plt.gca()

    n_patients = len(plot_df)
    x = np.arange(n_patients)
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        plot_df["actual_ratio"],
        width,
        color="lightcoral",
        edgecolor="red",
        label="实际原始细胞比例"
    )

    rects2 = ax.bar(
        x + width / 2,
        plot_df["predicted_ratio"],
        width,
        color="lightblue",
        edgecolor="blue",
        label="预测原始细胞比例"
    )

    def add_value_labels(rects, offset=0.02):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.,
                    height + offset,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    add_value_labels(rects1)
    add_value_labels(rects2)

    legend_elements = [
        Line2D([0], [0], color="lightcoral", lw=4, label="实际原始细胞比例"),
        Line2D([0], [0], color="lightblue", lw=4, label="预测原始细胞比例"),
    ]

    ax.set_ylabel("原始细胞比例", fontsize=12)
    ax.set_xlabel("患者 (按预测比例排序: HC → AML)", fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14, pad=20)

    xtick_labels = [
        f"{str(pid)[:20]}\n{t if pd.notna(t) else 'NA'}"
        for pid, t in zip(plot_df["patient_id"], plot_df["type"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

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

    print(f"✅ 图形已保存: {save_path}")
    if best_thresh > 0:
        print(f"📊 最佳区分阈值: {best_thresh:.4f}")

    return plot_df, best_thresh


# =========================
# 单任务处理
# =========================

def run_one_task(cell_result_csv, patient_info_xlsx, output_png, output_excel):
    """
    单个任务处理
    """
    print("\n" + "=" * 80)
    print(f"📦 开始处理: {cell_result_csv}")
    print("=" * 80)

    if not os.path.exists(cell_result_csv):
        print(f"❌ 找不到细胞结果 CSV: {cell_result_csv}")
        return

    if not os.path.exists(patient_info_xlsx):
        print(f"❌ 找不到患者信息 XLSX: {patient_info_xlsx}")
        return

    ensure_parent_dir(output_excel)
    ensure_parent_dir(output_png)

    print(f"📖 读取细胞结果: {cell_result_csv}")
    cell_df = pd.read_csv(cell_result_csv)

    print(f"📖 读取患者信息: {patient_info_xlsx}")
    patient_info_df = pd.read_excel(patient_info_xlsx)

    # 注意：这里不再强制要求 true_label
    required_cols_csv = ["image", "pred_label"]
    for col in required_cols_csv:
        if col not in cell_df.columns:
            print(f"❌ CSV 缺少必要列: {col}")
            return

    required_cols_xlsx = ["正式编号", "患者大类型"]
    for col in required_cols_xlsx:
        if col not in patient_info_df.columns:
            print(f"❌ XLSX 缺少必要列: {col}")
            return

    # 患者汇总
    patient_summary, cell_df_with_info = build_patient_summary(cell_df, patient_info_df)

    # 患者 × 细胞类型统计
    patient_celltype_detail, patient_celltype_wide = build_patient_celltype_stats(cell_df_with_info)

    # 额外补一个患者级“每类细胞数量汇总”
    patient_type_count_summary = (
        patient_celltype_detail
        .groupby("patient_id")
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

    # 保存到 Excel 多 sheet
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        patient_summary.to_excel(writer, sheet_name="patient_summary", index=False)
        patient_celltype_detail.to_excel(writer, sheet_name="patient_celltype_detail", index=False)
        patient_celltype_wide.to_excel(writer, sheet_name="patient_celltype_wide", index=False)
        patient_type_count_summary.to_excel(writer, sheet_name="patient_type_count_summary", index=False)

    print(f"✅ 患者统计结果已保存: {output_excel}")

    # 画图
    plot_df, best_thresh = plot_patient_ratios(patient_summary, output_png)

    print("\n===== 患者汇总预览 =====")
    print(patient_summary.head())

    print("\n===== 患者-细胞类型统计预览 =====")
    print(patient_celltype_detail.head())

    print("\n===== 患者-细胞类型宽表预览 =====")
    print(patient_celltype_wide.head())

    print("\n===== 患者级类型统计汇总预览 =====")
    print(patient_type_count_summary.head())


# =========================
# 主函数
# =========================

def main():
    set_chinese_font()

    patient_info_xlsx = "/root/autodl-tmp/data/样本信息整理260323.xlsx"

    tasks = [
        {
            "cell_result_csv": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/val_260323/val_results_20260324-200121.csv",
            "patient_info_xlsx": patient_info_xlsx,
            "output_png": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/val_260323/patient_ratio_from_cell_results.png",
            "output_excel": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/val_260323/patient_ratio_from_cell_results.xlsx",
        },

        {
            "cell_result_csv": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/train_260323/train_results_20260324-195949.csv",
            "patient_info_xlsx": patient_info_xlsx,
            "output_png": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/train_260323/patient_ratio_from_cell_results.png",
            "output_excel": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/train_260323/patient_ratio_from_cell_results.xlsx",
        },

        {
            "cell_result_csv": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_BJH_260323/val_results_20260324-195044.csv",
            "patient_info_xlsx": patient_info_xlsx,
            "output_png": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_BJH_260323/patient_ratio_from_cell_results.png",
            "output_excel": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_BJH_260323/patient_ratio_from_cell_results.xlsx",
        },

        {
            "cell_result_csv": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_FXH_noALL_260323/val_results_20260324-195123.csv",
            "patient_info_xlsx": patient_info_xlsx,
            "output_png": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_FXH_noALL_260323/patient_ratio_from_cell_results.png",
            "output_excel": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_FXH_noALL_260323/patient_ratio_from_cell_results.xlsx",
        },

        {
            "cell_result_csv": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_test/test_TJMU_260323/val_results_20260324-195015.csv",
            "patient_info_xlsx": patient_info_xlsx,
            "output_png": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_TJMU_260323/patient_ratio_from_cell_results.png",
            "output_excel": "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_patient/test_TJMU_260323/patient_ratio_from_cell_results.xlsx",
        },
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n🚀 批处理任务 {i}/{len(tasks)}")
        run_one_task(
            cell_result_csv=task["cell_result_csv"],
            patient_info_xlsx=task["patient_info_xlsx"],
            output_png=task["output_png"],
            output_excel=task["output_excel"],
        )


if __name__ == "__main__":
    main()