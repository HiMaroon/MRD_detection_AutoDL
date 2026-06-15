import argparse
from pathlib import Path
from typing import List

import pandas as pd

from fivefold_patient_ensemble_utils import (
    add_cell_predictions,
    best_patient_threshold_by_roc,
    build_patient_summary,
    discover_fold_csvs,
    evaluate_patient_predictions,
    load_patient_info,
    merge_fold_probabilities,
    parse_thresholds,
    plot_patient_ratios,
    set_chinese_font,
    write_dataframe_outputs,
)


def process_split(args, split_name: str, patient_info_df: pd.DataFrame | None, thresholds: List[float]):
    fold_csvs = discover_fold_csvs(args.root, split_name)
    if len(fold_csvs) == 0:
        print(f"⚠️ [{split_name}] 未发现 fold*/eval/{split_name}/val_results.csv，跳过")
        return None

    print(f"\n{'=' * 100}\n🚀 [{split_name}] 读取 {len(fold_csvs)} 个 fold 结果")
    for path in fold_csvs:
        print(f"   - {path}")

    merged_probs = merge_fold_probabilities(fold_csvs, keep_fold_prob_cols=args.keep_fold_prob_cols)
    split_out = args.out_root / split_name
    split_out.mkdir(parents=True, exist_ok=True)
    merged_probs.to_csv(split_out / "ensemble_mean_probabilities.csv", index=False)

    rows = []
    best_by_metric = None
    for threshold in thresholds:
        threshold_tag = f"thr_{threshold:.3f}".replace(".", "p")
        threshold_out = split_out / threshold_tag
        threshold_out.mkdir(parents=True, exist_ok=True)

        cell_df = add_cell_predictions(merged_probs, threshold)
        cell_df.to_csv(threshold_out / "val_results.csv", index=False)

        patient_summary, cell_detail = build_patient_summary(
            cell_df,
            patient_info_df,
            patient_prob_threshold=args.patient_prob_threshold,
            patient_hard_ratio_threshold=args.patient_hard_ratio_threshold,
        )
        best_hard_patient_threshold = best_patient_threshold_by_roc(patient_summary, "hard_predicted_ratio")
        if best_hard_patient_threshold is not None:
            patient_summary["pred_label_by_best_hard_ratio"] = (
                patient_summary["hard_predicted_ratio"] >= best_hard_patient_threshold
            ).astype(int)
        else:
            patient_summary["pred_label_by_best_hard_ratio"] = patient_summary["pred_label_by_hard_ratio"]

        patient_summary["cell_positive_threshold"] = threshold
        patient_summary["patient_hard_ratio_threshold"] = args.patient_hard_ratio_threshold
        patient_summary["patient_best_hard_ratio_threshold"] = best_hard_patient_threshold
        write_dataframe_outputs(
            patient_summary,
            threshold_out / "patient_summary.csv",
            threshold_out / "patient_summary.xlsx",
        )
        cell_detail.to_csv(threshold_out / "cell_results_with_patient_fields.csv", index=False)

        hard_metrics = evaluate_patient_predictions(
            patient_summary,
            pred_col="pred_label_by_hard_ratio",
            score_col="hard_predicted_ratio",
        )
        best_hard_metrics = evaluate_patient_predictions(
            patient_summary,
            pred_col="pred_label_by_best_hard_ratio",
            score_col="hard_predicted_ratio",
        )
        metric_row = {
            "split": split_name,
            "cell_positive_threshold": threshold,
            "patient_hard_ratio_threshold": args.patient_hard_ratio_threshold,
            "patient_best_hard_ratio_threshold": best_hard_patient_threshold,
            **{f"fixed_{k}": v for k, v in hard_metrics.items()},
            **{f"best_{k}": v for k, v in best_hard_metrics.items()},
        }
        rows.append(metric_row)

        if args.save_plots:
            plot_patient_ratios(
                patient_summary,
                threshold_out / "patient_ratio_hard_label.png",
                ratio_col="hard_predicted_ratio",
                title=f"{split_name} 5-fold ensemble 阈值={threshold:.3f}: 实际 vs 硬标签预测N/M比例",
                label="硬标签预测N/M比例",
            )

        selector = args.select_metric
        if selector in metric_row and pd.notna(metric_row[selector]):
            if best_by_metric is None or float(metric_row[selector]) > float(best_by_metric[selector]):
                best_by_metric = metric_row

    sweep_df = pd.DataFrame(rows)
    write_dataframe_outputs(
        sweep_df,
        split_out / "threshold_sweep_summary.csv",
        split_out / "threshold_sweep_summary.xlsx",
    )
    print(f"✅ [{split_name}] 阈值扫描汇总已保存: {split_out / 'threshold_sweep_summary.csv'}")

    if best_by_metric is not None:
        print(
            f"🏆 [{split_name}] 按 {args.select_metric} 最优 cell 阈值="
            f"{best_by_metric['cell_positive_threshold']:.3f}, {args.select_metric}={best_by_metric[args.select_metric]:.4f}"
        )
    return sweep_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于已跑完的5个fold结果做平均概率ensemble、细胞阳性阈值扫描，并输出患者级分级/诊断结果。"
    )
    parser.add_argument("--root", type=Path, required=True, help="包含 fold*/eval/<split>/val_results.csv 的5-fold根目录")
    parser.add_argument("--out-root", type=Path, default=None, help="输出目录，默认 <root>/threshold_sweep_ensemble")
    parser.add_argument("--splits", nargs="*", default=["val", "test_BJH", "test_FXH_noALL", "test_TJMU"], help="要处理的split")
    parser.add_argument("--thresholds", nargs="*", default=None, help="显式细胞阳性阈值，例如 0.5 0.6 0.7 或 0.5,0.6,0.7")
    parser.add_argument("--threshold-start", type=float, default=0.5)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--patient-hard-ratio-threshold", type=float, default=0.5, help="用硬标签患者比例分诊断时的患者级阈值")
    parser.add_argument("--patient-prob-threshold", type=float, default=0.5, help="保留在patient_summary中的概率均值分类阈值")
    parser.add_argument("--patient-info-xlsx", type=Path, default=Path("/root/autodl-tmp/data/patient_data_260416.xlsx"))
    parser.add_argument("--select-metric", default="fixed_specificity_hc", help="打印最优阈值时使用的汇总列名")
    parser.add_argument("--keep-fold-prob-cols", action="store_true", help="细胞级输出保留每个fold的概率列")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="不保存患者比例图")
    parser.set_defaults(save_plots=True)
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = args.root / "threshold_sweep_ensemble"
    return args


def main():
    args = parse_args()
    set_chinese_font()
    args.out_root.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds, args.threshold_start, args.threshold_end, args.threshold_step)
    patient_info_df = load_patient_info(args.patient_info_xlsx)

    all_summaries = []
    for split_name in args.splits:
        summary = process_split(args, split_name, patient_info_df, thresholds)
        if summary is not None:
            all_summaries.append(summary)

    if all_summaries:
        all_df = pd.concat(all_summaries, ignore_index=True)
        write_dataframe_outputs(
            all_df,
            args.out_root / "all_splits_threshold_sweep_summary.csv",
            args.out_root / "all_splits_threshold_sweep_summary.xlsx",
        )
        print(f"✅ 全部split阈值扫描汇总已保存: {args.out_root / 'all_splits_threshold_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
