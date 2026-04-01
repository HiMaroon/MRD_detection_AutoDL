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


def merge_one_split(root: Path, split_name: str, out_root: Path, binary_threshold: float):
    fold_csvs = discover_fold_csvs(root, split_name)
    if len(fold_csvs) == 0:
        print(f"⚠️ [{split_name}] 未找到 fold 结果，跳过")
        return

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

    if "true_label" in merged.columns:
        num_classes = None
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

    print(f"📄 [{split_name}] 已保存: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="融合 5-fold 外部测试预测结果")
    p.add_argument("--root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold", help="包含 fold*/eval 的根目录")
    p.add_argument("--out-root", default=None, help="输出目录，默认 <root>/ensemble_eval")
    p.add_argument("--splits", nargs="*", default=["test_BJH", "test_FXH_noALL", "test_TJMU"], help="待融合的外部测试集")
    p.add_argument("--binary-threshold", type=float, default=0.5, help="二分类 prob_class_1 阈值")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else (root / "ensemble_eval")

    for split_name in args.splits:
        merge_one_split(root, split_name, out_root, args.binary_threshold)


if __name__ == "__main__":
    main()
