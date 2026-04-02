import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.test import calculate_and_save_metrics
from tools.patient_analysis import run_one_task, set_chinese_font


def discover_fold_csvs(root: Path, split_name: str) -> List[Path]:
    return sorted(root.glob(f"fold*/eval/{split_name}/val_results.csv"))


def parse_patient_id_from_image(image_name: str) -> str | None:
    """
    从单细胞图像名中解析 patient_id。
    例: PKUPH-106-10_000_P2 -> PKUPH-106
    """
    image_name = str(image_name)
    prefix = image_name.split("_")[0]
    parts = prefix.split("-")
    if len(parts) < 3:
        return None
    return f"{parts[0]}-{parts[1]}"


def build_patient_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将单个 fold 的细胞级结果聚合为患者级结果，避免依赖细胞一一对应。
    """
    work = df.copy()
    if "image" not in work.columns:
        raise ValueError("输入结果缺少 image 列，无法解析 patient_id")

    work["patient_id"] = work["image"].apply(parse_patient_id_from_image)
    work = work[work["patient_id"].notna()].copy()
    if work.empty:
        raise ValueError("无法从 image 中解析任何 patient_id")

    prob_cols = [c for c in work.columns if c.startswith("prob_class_")]
    if prob_cols:
        agg_map = {c: "mean" for c in prob_cols}
        patient_df = work.groupby("patient_id", as_index=False).agg(agg_map)
        patient_df["pred_label"] = patient_df[prob_cols].values.argmax(axis=1)
    else:
        patient_df = work.groupby("patient_id", as_index=False).agg(
            pred_label=("pred_label", lambda x: int(pd.Series(x).mode().iloc[0]))
        )

    return patient_df


def merge_patients_across_folds(dfs: List[pd.DataFrame], binary_threshold: float) -> pd.DataFrame:
    """
    患者层面融合：
    - 先将每个 fold 的细胞级结果聚合为患者级
    - 再在患者维度融合 fold 结果
    """
    patient_dfs = [build_patient_level_df(df) for df in dfs]
    has_probs = all(any(c.startswith("prob_class_") for c in df.columns) for df in patient_dfs)

    merged = patient_dfs[0][["patient_id"]].copy()
    if has_probs:
        prob_cols = [c for c in patient_dfs[0].columns if c.startswith("prob_class_")]
        for i, df in enumerate(patient_dfs, start=1):
            sub = df[["patient_id"] + prob_cols].rename(columns={c: f"{c}_f{i}" for c in prob_cols})
            merged = merged.merge(sub, on="patient_id", how="inner")

        for c in prob_cols:
            fold_cols = [f"{c}_f{i}" for i in range(1, len(patient_dfs) + 1)]
            merged[c] = merged[fold_cols].mean(axis=1)

        if len(prob_cols) == 2:
            merged["pred_label"] = (merged["prob_class_1"] >= binary_threshold).astype(int)
        else:
            merged["pred_label"] = merged[prob_cols].values.argmax(axis=1)
        keep_cols = ["patient_id", "pred_label"] + prob_cols
    else:
        pred_cols = []
        for i, df in enumerate(patient_dfs, start=1):
            col = f"pred_label_f{i}"
            pred_cols.append(col)
            merged = merged.merge(df[["patient_id", "pred_label"]].rename(columns={"pred_label": col}), on="patient_id", how="inner")
        votes = merged[pred_cols].to_numpy(dtype=int)
        merged["pred_label"] = [int(np.bincount(row).argmax()) for row in votes]
        keep_cols = ["patient_id", "pred_label"]

    return merged[keep_cols].sort_values("patient_id").reset_index(drop=True)


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
        merged["correct"] = (merged["true_label"].astype(int) == merged["pred_label"].astype(int))

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
        merged["correct"] = (merged["true_label"].astype(int) == merged["pred_label"].astype(int))

    keep_cols = ["image"]
    if "true_label" in merged.columns:
        keep_cols += ["true_label"]
    keep_cols += ["pred_label"]
    if "correct" in merged.columns:
        keep_cols += ["correct"]
    return merged[keep_cols].sort_values("image").reset_index(drop=True)


def run_patient_level_analysis(out_csv: Path, patient_xlsx: Path, out_dir: Path):
    """
    参考第一个代码中的 run_eval_and_patient_report 流程，
    对融合后的细胞级结果做患者级分析。
    """
    patient_out_dir = out_dir / "patient_report"
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    set_chinese_font()
    run_one_task(
        cell_result_csv=str(out_csv),
        patient_info_xlsx=str(patient_xlsx),
        output_png=str(patient_out_dir / "patient_ratio_from_cell_results.png"),
        output_excel=str(patient_out_dir / "patient_ratio_from_cell_results.xlsx"),
    )
    print(f"📊 患者级结果已保存到: {patient_out_dir}")


def merge_one_split(
    root: Path,
    split_name: str,
    out_root: Path,
    binary_threshold: float,
    patient_xlsx: Path | None = None,
    ensemble_level: str = "cell",
):
    fold_csvs = discover_fold_csvs(root, split_name)
    if len(fold_csvs) == 0:
        print(f"⚠️ [{split_name}] 未找到 fold 结果，跳过")
        return

    dfs = [pd.read_csv(p) for p in fold_csvs]
    if ensemble_level == "patient":
        merged_patient = merge_patients_across_folds(dfs, binary_threshold=binary_threshold)
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "patient_ensemble_results.csv"
        merged_patient.to_csv(out_csv, index=False)
        print(f"✅ [{split_name}] 患者层面融合完成: {out_csv}")

        if patient_xlsx is not None and patient_xlsx.exists():
            info = pd.read_excel(patient_xlsx)
            if "正式编号" in info.columns and "患者大类型" in info.columns:
                info["正式编号"] = info["正式编号"].astype(str).str.strip()
                out_eval = merged_patient.merge(
                    info[["正式编号", "患者大类型"]].drop_duplicates(),
                    left_on="patient_id",
                    right_on="正式编号",
                    how="left",
                )
                out_eval.to_csv(out_dir / "patient_ensemble_with_type.csv", index=False)
                print(f"📄 [{split_name}] 已保存患者层面融合+标签结果")
        return

    has_probs = all(any(c.startswith("prob_class_") for c in df.columns) for df in dfs)
    if has_probs:
        merged = merge_with_probs(dfs, binary_threshold=binary_threshold)
        print(f"✅ [{split_name}] 细胞层面使用概率融合（平均概率）")
    else:
        merged = merge_with_vote(dfs)
        print(f"✅ [{split_name}] 细胞层面无概率列，使用多数投票融合")

    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "val_results.csv"
    merged.to_csv(out_csv, index=False)

    # 细胞级指标
    if "true_label" in merged.columns:
        prob_cols = [c for c in merged.columns if c.startswith("prob_class_")]
        if len(prob_cols) > 0:
            num_classes = len(prob_cols)
        else:
            num_classes = int(merged["pred_label"].max()) + 1

        calculate_and_save_metrics(
            df=merged,
            output_dir=str(out_dir),
            split="val",
            timestamp="ensemble",
            num_classes=num_classes,
        )

    print(f"📄 [{split_name}] 已保存细胞级结果: {out_csv}")

    # 患者级分析
    if patient_xlsx is not None and patient_xlsx.exists():
        try:
            run_patient_level_analysis(
                out_csv=out_csv,
                patient_xlsx=patient_xlsx,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"⚠️ [{split_name}] 患者级分析失败: {e}")
    else:
        print(f"ℹ️ [{split_name}] 未提供有效 patient_xlsx，跳过患者级分析")


def parse_args():
    p = argparse.ArgumentParser(description="融合 5-fold 外部测试预测结果，并补充患者层面验证")
    p.add_argument("--root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold", help="包含 fold*/eval 的根目录")
    p.add_argument("--out-root", default=None, help="输出目录，默认 <root>/ensemble_eval")
    p.add_argument("--splits", nargs="*", default=["test_BJH", "test_FXH_noALL", "test_TJMU"], help="待融合的外部测试集")
    p.add_argument("--binary-threshold", type=float, default=0.5, help="二分类 prob_class_1 阈值")
    p.add_argument(
        "--ensemble-level",
        choices=["cell", "patient"],
        default="patient",
        help="融合层级: cell(细胞级，需要可对齐) / patient(患者级，不要求细胞一一对应)",
    )
    p.add_argument("--patient-xlsx", default="/root/autodl-tmp/data/patient_data_260323.xlsx", help="患者信息表，用于患者层面分析")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else (root / "ensemble_eval")
    patient_xlsx = Path(args.patient_xlsx) if args.patient_xlsx else None

    for split_name in args.splits:
        merge_one_split(
            root=root,
            split_name=split_name,
            out_root=out_root,
            binary_threshold=args.binary_threshold,
            patient_xlsx=patient_xlsx,
            ensemble_level=args.ensemble_level,
        )


if __name__ == "__main__":
    main()

# import argparse
# from pathlib import Path
# from typing import List

# import numpy as np
# import pandas as pd

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# import sys
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from tools.test import calculate_and_save_metrics


# def discover_fold_csvs(root: Path, split_name: str) -> List[Path]:
#     return sorted(root.glob(f"fold*/eval/{split_name}/val_results.csv"))


# def merge_with_probs(dfs: List[pd.DataFrame], binary_threshold: float) -> pd.DataFrame:
#     prob_cols = [c for c in dfs[0].columns if c.startswith("prob_class_")]
#     num_classes = len(prob_cols)

#     merged = dfs[0][["image"]].copy()
#     if "true_label" in dfs[0].columns:
#         merged["true_label"] = dfs[0]["true_label"].values

#     for i, df in enumerate(dfs, start=1):
#         sub = df[["image"] + prob_cols].copy()
#         sub = sub.rename(columns={c: f"{c}_f{i}" for c in prob_cols})
#         merged = merged.merge(sub, on="image", how="inner")

#     for c in prob_cols:
#         fold_cols = [f"{c}_f{i}" for i in range(1, len(dfs) + 1)]
#         merged[c] = merged[fold_cols].mean(axis=1)

#     if num_classes == 2:
#         merged["pred_label"] = (merged["prob_class_1"] >= binary_threshold).astype(int)
#     else:
#         merged["pred_label"] = merged[prob_cols].values.argmax(axis=1)

#     if "true_label" in merged.columns:
#         merged["correct"] = (merged["true_label"].astype(int) == merged["pred_label"].astype(int))

#     keep_cols = ["image"]
#     if "true_label" in merged.columns:
#         keep_cols += ["true_label"]
#     keep_cols += ["pred_label"]
#     if "correct" in merged.columns:
#         keep_cols += ["correct"]
#     keep_cols += prob_cols
#     return merged[keep_cols].sort_values("image").reset_index(drop=True)


# def merge_with_vote(dfs: List[pd.DataFrame]) -> pd.DataFrame:
#     merged = dfs[0][["image"]].copy()
#     if "true_label" in dfs[0].columns:
#         merged["true_label"] = dfs[0]["true_label"].values

#     pred_cols = []
#     for i, df in enumerate(dfs, start=1):
#         col = f"pred_label_f{i}"
#         pred_cols.append(col)
#         sub = df[["image", "pred_label"]].rename(columns={"pred_label": col})
#         merged = merged.merge(sub, on="image", how="inner")

#     votes = merged[pred_cols].to_numpy(dtype=int)
#     majority = []
#     for row in votes:
#         labels, counts = np.unique(row, return_counts=True)
#         majority.append(int(labels[np.argmax(counts)]))

#     merged["pred_label"] = majority
#     if "true_label" in merged.columns:
#         merged["correct"] = (merged["true_label"].astype(int) == merged["pred_label"].astype(int))

#     keep_cols = ["image"]
#     if "true_label" in merged.columns:
#         keep_cols += ["true_label"]
#     keep_cols += ["pred_label"]
#     if "correct" in merged.columns:
#         keep_cols += ["correct"]
#     return merged[keep_cols].sort_values("image").reset_index(drop=True)


# def merge_one_split(root: Path, split_name: str, out_root: Path, binary_threshold: float):
#     fold_csvs = discover_fold_csvs(root, split_name)
#     if len(fold_csvs) == 0:
#         print(f"⚠️ [{split_name}] 未找到 fold 结果，跳过")
#         return

#     dfs = [pd.read_csv(p) for p in fold_csvs]
#     has_probs = all(any(c.startswith("prob_class_") for c in df.columns) for df in dfs)

#     if has_probs:
#         merged = merge_with_probs(dfs, binary_threshold=binary_threshold)
#         print(f"✅ [{split_name}] 使用概率融合（平均概率）")
#     else:
#         merged = merge_with_vote(dfs)
#         print(f"✅ [{split_name}] 无概率列，使用多数投票融合")

#     out_dir = out_root / split_name
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_csv = out_dir / "val_results.csv"
#     merged.to_csv(out_csv, index=False)

#     if "true_label" in merged.columns:
#         num_classes = None
#         prob_cols = [c for c in merged.columns if c.startswith("prob_class_")]
#         if len(prob_cols) > 0:
#             num_classes = len(prob_cols)
#         else:
#             num_classes = int(merged["pred_label"].max()) + 1

#         calculate_and_save_metrics(
#             df=merged,
#             output_dir=str(out_dir),
#             split="val",
#             timestamp="ensemble",
#             num_classes=num_classes,
#         )

#     print(f"📄 [{split_name}] 已保存: {out_csv}")


# def parse_args():
#     p = argparse.ArgumentParser(description="融合 5-fold 外部测试预测结果")
#     p.add_argument("--root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold", help="包含 fold*/eval 的根目录")
#     p.add_argument("--out-root", default=None, help="输出目录，默认 <root>/ensemble_eval")
#     p.add_argument("--splits", nargs="*", default=["test_BJH", "test_FXH_noALL", "test_TJMU"], help="待融合的外部测试集")
#     p.add_argument("--binary-threshold", type=float, default=0.5, help="二分类 prob_class_1 阈值")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     root = Path(args.root)
#     out_root = Path(args.out_root) if args.out_root else (root / "ensemble_eval")

#     for split_name in args.splits:
#         merge_one_split(root, split_name, out_root, args.binary_threshold)


# if __name__ == "__main__":
#     main()
