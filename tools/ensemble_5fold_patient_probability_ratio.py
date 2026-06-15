import argparse
from pathlib import Path

import pandas as pd

from fivefold_patient_ensemble_utils import (
    add_cell_predictions,
    build_patient_summary,
    discover_fold_csvs,
    evaluate_patient_predictions,
    load_patient_info,
    merge_fold_probabilities,
    plot_patient_ratios,
    set_chinese_font,
    write_dataframe_outputs,
)


def process_split(args, split_name: str, patient_info_df: pd.DataFrame | None):
    fold_csvs = discover_fold_csvs(args.root, split_name)
    if len(fold_csvs) == 0:
        print(f"⚠️ [{split_name}] 未发现 fold*/eval/{split_name}/val_results.csv，跳过")
        return None

    print(f"\n{'=' * 100}\n🚀 [{split_name}] 读取 {len(fold_csvs)} 个 fold 结果")
    for path in fold_csvs:
        print(f"   - {path}")

    out_dir = args.out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_probs = merge_fold_probabilities(fold_csvs, keep_fold_prob_cols=args.keep_fold_prob_cols)
    cell_df = add_cell_predictions(merged_probs, args.cell_positive_threshold)
    cell_df.to_csv(out_dir / "val_results.csv", index=False)

    patient_summary, cell_detail = build_patient_summary(
        cell_df,
        patient_info_df,
        patient_prob_threshold=args.patient_prob_threshold,
        patient_hard_ratio_threshold=args.patient_hard_ratio_threshold,
    )
    write_dataframe_outputs(
        patient_summary,
        out_dir / "patient_summary_hard_and_prob.csv",
        out_dir / "patient_summary_hard_and_prob.xlsx",
    )
    cell_detail.to_csv(out_dir / "cell_results_with_patient_fields.csv", index=False)

    hard_metrics = evaluate_patient_predictions(
        patient_summary,
        pred_col="pred_label_by_hard_ratio",
        score_col="hard_predicted_ratio",
    )
    prob_metrics = evaluate_patient_predictions(
        patient_summary,
        pred_col="pred_label_by_mean_prob",
        score_col="mean_prob_class_1",
    )
    metrics = {
        "split": split_name,
        "cell_positive_threshold_for_hard_ratio": args.cell_positive_threshold,
        "patient_hard_ratio_threshold": args.patient_hard_ratio_threshold,
        "patient_prob_threshold": args.patient_prob_threshold,
        **{f"hard_ratio_{k}": v for k, v in hard_metrics.items()},
        **{f"mean_prob_{k}": v for k, v in prob_metrics.items()},
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "patient_classification_metrics.csv", index=False)

    if args.save_plots:
        plot_patient_ratios(
            patient_summary,
            out_dir / "patient_ratio_hard_label.png",
            ratio_col="hard_predicted_ratio",
            title=f"{split_name}: 实际 vs 硬标签预测N/M比例",
            label="硬标签预测N/M比例",
        )
        plot_patient_ratios(
            patient_summary,
            out_dir / "patient_ratio_mean_probability.png",
            ratio_col="mean_prob_class_1",
            title=f"{split_name}: 实际N/M比例 vs 平均prob_class_1",
            label="prob_class_1均值",
        )

    print(f"✅ [{split_name}] 患者硬标签比例 + 概率均值结果已保存: {out_dir}")
    return metrics_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于已跑完的5个fold结果做平均概率ensemble，同时输出患者硬标签比例和prob_class_1均值，并按概率均值做患者分类。"
    )
    parser.add_argument("--root", type=Path, required=True, help="包含 fold*/eval/<split>/val_results.csv 的5-fold根目录")
    parser.add_argument("--out-root", type=Path, default=None, help="输出目录，默认 <root>/patient_probability_ratio_ensemble")
    parser.add_argument("--splits", nargs="*", default=["val", "test_BJH", "test_FXH_noALL", "test_TJMU"], help="要处理的split")
    parser.add_argument("--cell-positive-threshold", type=float, default=0.5, help="生成硬标签比例时使用的细胞级prob_class_1阈值")
    parser.add_argument("--patient-hard-ratio-threshold", type=float, default=0.5, help="用硬标签患者比例分类时的患者级阈值")
    parser.add_argument("--patient-prob-threshold", type=float, default=0.5, help="用患者prob_class_1均值分类时的患者级阈值")
    parser.add_argument("--patient-info-xlsx", type=Path, default=Path("/root/autodl-tmp/data/patient_data_260416.xlsx"))
    parser.add_argument("--keep-fold-prob-cols", action="store_true", help="细胞级输出保留每个fold的概率列")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="不保存患者比例图")
    parser.set_defaults(save_plots=True)
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = args.root / "patient_probability_ratio_ensemble"
    return args


def main():
    args = parse_args()
    set_chinese_font()
    args.out_root.mkdir(parents=True, exist_ok=True)
    patient_info_df = load_patient_info(args.patient_info_xlsx)

    all_metrics = []
    for split_name in args.splits:
        metrics = process_split(args, split_name, patient_info_df)
        if metrics is not None:
            all_metrics.append(metrics)

    if all_metrics:
        all_df = pd.concat(all_metrics, ignore_index=True)
        write_dataframe_outputs(
            all_df,
            args.out_root / "all_splits_patient_classification_metrics.csv",
            args.out_root / "all_splits_patient_classification_metrics.xlsx",
        )
        print(f"✅ 全部split患者分类指标已保存: {args.out_root / 'all_splits_patient_classification_metrics.csv'}")


if __name__ == "__main__":
    main()
