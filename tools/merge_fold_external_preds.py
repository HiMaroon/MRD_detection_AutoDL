# import argparse
# import json
# import os
# import re
# from pathlib import Path
# from typing import List, Optional, Tuple

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import numpy as np
# import pandas as pd
# from matplotlib.lines import Line2D
# from sklearn.metrics import (
#     accuracy_score,
#     auc,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     roc_curve,
# )

# # =========================
# # 全局配置
# # =========================

# font_path = "/root/autodl-tmp/projects/myq/SingleCellProject/tools/MSYH.TTC"
# FONT_NAME = "MSYH.TTC"
# POSITIVE_CLASS = 1

# # 参考用户给的 patient_analysis 风格代码
# cell_dict_big = {
#     "V": 0, "0": 0,
#     "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
#     "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
#     "E": 2, "B": 2, "E1": 2, "B1": 2,
#     "M0": 2, "M2": 2, "R2": 2, "R3": 2,
#     "J2": 2, "J3": 2, "J4": 2,
#     "P": 2, "P1": 2, "P2": 2, "P3": 2,
#     "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
# }


# # =========================
# # 基础工具
# # =========================

# def ensure_dir(path: Path):
#     path.mkdir(parents=True, exist_ok=True)


# def ensure_parent_dir(file_path: Path):
#     file_path.parent.mkdir(parents=True, exist_ok=True)


# def set_chinese_font():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     candidates = [
#         font_path,
#         os.path.join(current_dir, FONT_NAME),
#         os.path.join(os.getcwd(), FONT_NAME),
#     ]

#     selected_font = None
#     for path in candidates:
#         if path and os.path.exists(path):
#             selected_font = path
#             break

#     if selected_font:
#         try:
#             fm.fontManager.addfont(selected_font)
#             font_prop = fm.FontProperties(fname=selected_font)
#             font_name = font_prop.get_name()
#             plt.rcParams["font.family"] = font_name
#         except Exception:
#             plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
#     else:
#         plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]

#     plt.rcParams["axes.unicode_minus"] = False


# def normalize_patient_type(x):
#     if pd.isna(x):
#         return None
#     x = str(x).strip().upper()
#     if x in ["HC", "HD", "NORMAL", "HEALTHY"]:
#         return "HC"
#     if x == "AML":
#         return "AML"
#     return x


# def label_to_int(x):
#     x = normalize_patient_type(x)
#     if x == "HC":
#         return 0
#     if x == "AML":
#         return 1
#     return None


# # =========================
# # 参考文件中的命名解析逻辑
# # =========================

# def parse_image_info(image_name: str) -> Tuple[Optional[str], Optional[str]]:
#     """
#     参考用户提供代码:
#     PKUPH-106-10_000_P2
#     -> patient_id = PKUPH-106
#     -> smear_id   = 10

#     对 FXH-1_xxx 这类，也尽量兼容:
#     FXH-1_000_P2 -> patient_id=FXH-1, smear_id=None
#     """
#     image_name = str(image_name)
#     stem = Path(image_name).stem
#     prefix = stem.split("_")[0]
#     parts = prefix.split("-")

#     if len(parts) >= 3:
#         patient_id = f"{parts[0]}-{parts[1]}"
#         smear_id = parts[2]
#         return patient_id, smear_id
#     elif len(parts) >= 2:
#         patient_id = f"{parts[0]}-{parts[1]}"
#         return patient_id, None
#     return None, None


# def parse_cell_type(image_name: str) -> Optional[str]:
#     """
#     参考用户提供代码:
#     PKUPH-106-10_000_P2 -> P2
#     """
#     if pd.isna(image_name):
#         return None
#     image_name = str(image_name).strip()
#     stem = Path(image_name).stem
#     if "_" not in stem:
#         return None
#     return stem.split("_")[-1].strip().upper()


# def map_cell_type_to_binary(cell_type: str) -> int:
#     """
#     value == 1 -> 1
#     value == 0 / 2 / 不在字典中 -> 0
#     """
#     if pd.isna(cell_type) or cell_type is None:
#         return 0
#     v = cell_dict_big.get(str(cell_type).strip().upper(), 0)
#     return 1 if v == 1 else 0


# # =========================
# # fold结果发现
# # =========================

# def discover_fold_csvs(root: Path, split_name: str) -> List[Path]:
#     return sorted(root.glob(f"fold*/eval/{split_name}/val_results.csv"))


# def parse_fold_id(path: Path) -> int:
#     for part in path.parts:
#         m = re.fullmatch(r"fold(\d+)", part)
#         if m:
#             return int(m.group(1))
#     return -1


# # =========================
# # 患者表读取
# # =========================

# def load_patient_info(patient_xlsx: Optional[Path], sheet_name: Optional[str] = None) -> pd.DataFrame:
#     if patient_xlsx is None or not patient_xlsx.exists():
#         return pd.DataFrame(columns=["正式编号", "患者大类型", "true_label"])

#     if sheet_name:
#         df = pd.read_excel(patient_xlsx, sheet_name=sheet_name)
#     else:
#         df = pd.read_excel(patient_xlsx)

#     id_col = None
#     label_col = None

#     for c in df.columns:
#         c_str = str(c).strip()
#         if c_str in ["正式编号", "患者编号", "病人编号", "样本编号", "编号"]:
#             id_col = c
#         if c_str in ["患者大类型", "大类型", "标签", "分组", "patient_type"]:
#             label_col = c

#     if id_col is None:
#         raise ValueError(f"患者信息表中未找到患者ID列，现有列名：{list(df.columns)}")
#     if label_col is None:
#         raise ValueError(f"患者信息表中未找到患者标签列，现有列名：{list(df.columns)}")

#     out = df[[id_col, label_col]].copy()
#     out.columns = ["正式编号", "患者大类型"]
#     out["正式编号"] = out["正式编号"].astype(str).str.strip()
#     out["患者大类型"] = out["患者大类型"].apply(normalize_patient_type)
#     out["true_label"] = out["患者大类型"].apply(label_to_int)
#     out = out.dropna(subset=["正式编号"]).drop_duplicates(subset=["正式编号"]).reset_index(drop=True)
#     return out


# # =========================
# # 单fold：细胞级 -> 患者级
# # =========================

# def aggregate_one_fold_to_patient_level(
#     df: pd.DataFrame,
#     fold_id: int,
#     positive_class: int = 1,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     返回:
#     1) patient_df: 该fold的患者级汇总
#     2) cell_df: 补充了解析字段后的细胞表
#     """
#     if "image" not in df.columns:
#         raise ValueError("val_results.csv 中缺少 image 列")
#     if "pred_label" not in df.columns:
#         raise ValueError("val_results.csv 中缺少 pred_label 列")

#     df = df.copy()

#     parsed = df["image"].apply(parse_image_info)
#     df["patient_id"] = parsed.apply(lambda x: x[0])
#     df["smear_id"] = parsed.apply(lambda x: x[1])

#     df["cell_type"] = df["image"].apply(parse_cell_type)
#     df["mapped_label"] = df["cell_type"].apply(map_cell_type_to_binary)

#     df = df[df["patient_id"].notna()].copy()
#     if len(df) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     df["actual_positive"] = (df["mapped_label"] == positive_class).astype(int)
#     df["pred_positive"] = (df["pred_label"].astype(int) == positive_class).astype(int)
#     df["is_correct"] = (df["mapped_label"].astype(int) == df["pred_label"].astype(int)).astype(int)

#     has_prob1 = "prob_class_1" in df.columns

#     if has_prob1:
#         patient_df = (
#             df.groupby("patient_id")
#             .agg(
#                 fold_id=("patient_id", lambda _: fold_id),
#                 n_cells=("image", "count"),
#                 n_smears=("smear_id", "nunique"),
#                 actual_ratio=("actual_positive", "mean"),
#                 predicted_ratio=("pred_positive", "mean"),
#                 mean_prob_class_1=("prob_class_1", "mean"),
#                 accuracy=("is_correct", "mean"),
#                 pred_pos_cells=("pred_positive", "sum"),
#                 actual_pos_cells=("actual_positive", "sum"),
#             )
#             .reset_index()
#         )
#     else:
#         patient_df = (
#             df.groupby("patient_id")
#             .agg(
#                 fold_id=("patient_id", lambda _: fold_id),
#                 n_cells=("image", "count"),
#                 n_smears=("smear_id", "nunique"),
#                 actual_ratio=("actual_positive", "mean"),
#                 predicted_ratio=("pred_positive", "mean"),
#                 mean_prob_class_1=("pred_positive", "mean"),
#                 accuracy=("is_correct", "mean"),
#                 pred_pos_cells=("pred_positive", "sum"),
#                 actual_pos_cells=("actual_positive", "sum"),
#             )
#             .reset_index()
#         )

#     return patient_df.sort_values("patient_id").reset_index(drop=True), df


# # =========================
# # 跨fold患者级ensemble
# # =========================

# def patient_level_ensemble(
#     patient_fold_df: pd.DataFrame,
#     score_mode: str = "mean_prob",
#     threshold: float = 0.5,
#     min_folds_required: int = 1,
# ) -> pd.DataFrame:
#     if len(patient_fold_df) == 0:
#         return pd.DataFrame()

#     rows = []
#     for patient_id, sub in patient_fold_df.groupby("patient_id"):
#         sub = sub.sort_values("fold_id").reset_index(drop=True)

#         n_folds_available = len(sub)
#         if n_folds_available < min_folds_required:
#             continue

#         n_cells_sum = float(sub["n_cells"].sum())
#         n_cells_mean = float(sub["n_cells"].mean())

#         actual_ratio_mean = float(sub["actual_ratio"].mean())
#         actual_ratio_std = float(sub["actual_ratio"].std(ddof=0)) if len(sub) > 1 else 0.0

#         predicted_ratio_mean = float(sub["predicted_ratio"].mean())
#         predicted_ratio_std = float(sub["predicted_ratio"].std(ddof=0)) if len(sub) > 1 else 0.0

#         mean_prob_available = "mean_prob_class_1" in sub.columns and sub["mean_prob_class_1"].notna().any()

#         if mean_prob_available:
#             ensemble_mean_prob = float(sub["mean_prob_class_1"].mean())
#             ensemble_mean_prob_std = float(sub["mean_prob_class_1"].std(ddof=0)) if len(sub) > 1 else 0.0
#             weighted_mean_prob = float(np.average(sub["mean_prob_class_1"], weights=sub["n_cells"]))
#         else:
#             ensemble_mean_prob = np.nan
#             ensemble_mean_prob_std = np.nan
#             weighted_mean_prob = np.nan

#         weighted_predicted_ratio = float(np.average(sub["predicted_ratio"], weights=sub["n_cells"]))
#         weighted_actual_ratio = float(np.average(sub["actual_ratio"], weights=sub["n_cells"]))

#         votes = (sub["predicted_ratio"] >= 0.5).astype(int).tolist()
#         if len(votes) > 0:
#             labels, counts = np.unique(votes, return_counts=True)
#             ensemble_vote = int(labels[np.argmax(counts)])
#         else:
#             ensemble_vote = np.nan

#         if score_mode == "mean_prob":
#             final_score = ensemble_mean_prob if not pd.isna(ensemble_mean_prob) else predicted_ratio_mean
#         elif score_mode == "ratio":
#             final_score = predicted_ratio_mean
#         elif score_mode == "vote":
#             final_score = float(ensemble_vote) if not pd.isna(ensemble_vote) else np.nan
#         else:
#             raise ValueError(f"未知 score_mode: {score_mode}")

#         final_pred = int(final_score >= threshold) if not pd.isna(final_score) else np.nan

#         row = {
#             "patient_id": patient_id,
#             "n_folds_available": n_folds_available,
#             "n_cells_sum": n_cells_sum,
#             "n_cells_mean": n_cells_mean,
#             "actual_ratio": actual_ratio_mean,
#             "actual_ratio_std": actual_ratio_std,
#             "predicted_ratio": predicted_ratio_mean,
#             "predicted_ratio_std": predicted_ratio_std,
#             "ensemble_mean_prob_class_1": ensemble_mean_prob,
#             "ensemble_mean_prob_class_1_std": ensemble_mean_prob_std,
#             "weighted_mean_prob_class_1": weighted_mean_prob,
#             "weighted_predicted_ratio": weighted_predicted_ratio,
#             "weighted_actual_ratio": weighted_actual_ratio,
#             "ensemble_vote": ensemble_vote,
#             "final_score": final_score,
#             "final_pred": final_pred,
#         }

#         for _, r in sub.iterrows():
#             f = int(r["fold_id"])
#             row[f"fold{f}_n_cells"] = r.get("n_cells", np.nan)
#             row[f"fold{f}_actual_ratio"] = r.get("actual_ratio", np.nan)
#             row[f"fold{f}_predicted_ratio"] = r.get("predicted_ratio", np.nan)
#             row[f"fold{f}_mean_prob_class_1"] = r.get("mean_prob_class_1", np.nan)

#         rows.append(row)

#     return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


# # =========================
# # 绘图：参考用户给的 patient_analysis 风格
# # =========================

# def plot_patient_ratios(plot_df: pd.DataFrame, save_path: Path, title="各患者原始细胞比例对比 (实际 vs 预测)"):
#     target_types = ["AML", "HC"]
#     if "type" in plot_df.columns and plot_df["type"].isin(target_types).any():
#         plot_df = plot_df[plot_df["type"].isin(target_types)].copy()
#     else:
#         plot_df = plot_df.copy()

#     best_thresh = 0
#     if "type" in plot_df.columns and "AML" in plot_df["type"].unique() and "HC" in plot_df["type"].unique():
#         try:
#             y_true = (plot_df["type"] == "AML").astype(int)
#             y_score = plot_df["predicted_ratio"]
#             fpr, tpr, thresholds = roc_curve(y_true, y_score)
#             if len(thresholds) > 0:
#                 J = tpr - fpr
#                 best_thresh = thresholds[np.argmax(J)]
#         except Exception as e:
#             print(f"⚠️ 阈值计算失败: {e}")

#     plot_df = plot_df.sort_values("predicted_ratio", ascending=True).reset_index(drop=True)
#     plot_df["color"] = plot_df["type"].map({"AML": "red", "HC": "blue"}).fillna("black") if "type" in plot_df.columns else "black"

#     plt.figure(figsize=(max(18, len(plot_df) * 0.7), 9))
#     ax = plt.gca()

#     n_patients = len(plot_df)
#     x = np.arange(n_patients)
#     width = 0.35

#     rects1 = ax.bar(
#         x - width / 2,
#         plot_df["actual_ratio"],
#         width,
#         color="lightcoral",
#         edgecolor="red",
#         label="实际原始细胞比例"
#     )

#     rects2 = ax.bar(
#         x + width / 2,
#         plot_df["predicted_ratio"],
#         width,
#         color="lightblue",
#         edgecolor="blue",
#         label="预测原始细胞比例"
#     )

#     def add_value_labels(rects, offset=0.02):
#         for rect in rects:
#             height = rect.get_height()
#             if pd.notna(height) and height > 0:
#                 ax.text(
#                     rect.get_x() + rect.get_width() / 2.,
#                     height + offset,
#                     f"{height:.2f}",
#                     ha="center",
#                     va="bottom",
#                     fontsize=8
#                 )

#     add_value_labels(rects1)
#     add_value_labels(rects2)

#     legend_elements = [
#         Line2D([0], [0], color="lightcoral", lw=4, label="实际原始细胞比例"),
#         Line2D([0], [0], color="lightblue", lw=4, label="预测原始细胞比例"),
#     ]

#     ax.set_ylabel("原始细胞比例", fontsize=12)
#     ax.set_xlabel("患者 (按预测比例排序)", fontsize=12, labelpad=20)
#     ax.set_title(title, fontsize=14, pad=20)

#     xtick_labels = [
#         f"{str(pid)[:20]}\n{t if pd.notna(t) else 'NA'}"
#         for pid, t in zip(plot_df["patient_id"], plot_df.get("type", pd.Series(["NA"] * len(plot_df))))
#     ]
#     ax.set_xticks(x)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

#     if "type" in plot_df.columns:
#         for lbl, t in zip(ax.get_xticklabels(), plot_df["type"]):
#             if t == "AML":
#                 lbl.set_color("red")
#             elif t == "HC":
#                 lbl.set_color("blue")

#     ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc="upper right")
#     plt.grid(axis="y", linestyle="--", alpha=0.7)

#     plt.subplots_adjust(bottom=0.28)
#     plt.tight_layout()

#     plt.savefig(save_path, bbox_inches="tight", dpi=300)
#     plt.close()

#     print(f"✅ 图形已保存: {save_path}")
#     if best_thresh > 0:
#         print(f"📊 最佳区分阈值: {best_thresh:.4f}")

#     return plot_df, best_thresh


# def save_confusion_matrix_figure(cm: np.ndarray, out_path: Path):
#     plt.figure(figsize=(5, 4))
#     plt.imshow(cm, interpolation="nearest")
#     plt.title("Patient-level Confusion Matrix")
#     plt.colorbar()
#     tick_marks = np.arange(2)
#     plt.xticks(tick_marks, ["HC(0)", "AML(1)"])
#     plt.yticks(tick_marks, ["HC(0)", "AML(1)"])
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")

#     thresh = cm.max() / 2.0 if cm.size > 0 else 0
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(
#                 j, i, format(cm[i, j], "d"),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black"
#             )
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close()


# def save_roc_figure(fpr, tpr, roc_auc_val, out_path: Path):
#     plt.figure(figsize=(5, 5))
#     plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
#     plt.plot([0, 1], [0, 1], linestyle="--")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Patient-level ROC")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close()


# # =========================
# # 指标
# # =========================

# def compute_patient_metrics(df: pd.DataFrame, out_dir: Path, threshold: float):
#     ensure_dir(out_dir)

#     if "true_label" not in df.columns or df["true_label"].isna().all():
#         (out_dir / "patient_metrics.txt").write_text(
#             "未提供有效 true_label，跳过患者级指标计算。\n",
#             encoding="utf-8"
#         )
#         return

#     eval_df = df[df["true_label"].notna() & df["final_pred"].notna()].copy()
#     if len(eval_df) == 0:
#         (out_dir / "patient_metrics.txt").write_text(
#             "无可用于计算患者级指标的数据。\n",
#             encoding="utf-8"
#         )
#         return

#     y_true = eval_df["true_label"].astype(int).values
#     y_pred = eval_df["final_pred"].astype(int).values
#     y_score = eval_df["final_score"].astype(float).values

#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)

#     try:
#         auc_value = roc_auc_score(y_true, y_score)
#     except Exception:
#         auc_value = np.nan

#     cm = confusion_matrix(y_true, y_pred)
#     report = classification_report(y_true, y_pred, digits=4, zero_division=0)

#     text = []
#     text.append("Patient-level Metrics")
#     text.append("=" * 60)
#     text.append(f"n_patients = {len(eval_df)}")
#     text.append(f"threshold = {threshold}")
#     text.append(f"accuracy = {acc:.6f}")
#     text.append(f"precision = {prec:.6f}")
#     text.append(f"recall = {rec:.6f}")
#     text.append(f"f1 = {f1:.6f}")
#     text.append(f"roc_auc = {auc_value:.6f}" if not pd.isna(auc_value) else "roc_auc = NaN")
#     text.append("")
#     text.append("Confusion Matrix:")
#     text.append(str(cm))
#     text.append("")
#     text.append("Classification Report:")
#     text.append(report)

#     (out_dir / "patient_metrics.txt").write_text("\n".join(text), encoding="utf-8")

#     save_confusion_matrix_figure(cm, out_dir / "patient_confusion_matrix.png")

#     if len(np.unique(y_true)) == 2:
#         try:
#             fpr, tpr, _ = roc_curve(y_true, y_score)
#             roc_auc_val = auc(fpr, tpr)
#             save_roc_figure(fpr, tpr, roc_auc_val, out_dir / "patient_roc.png")
#         except Exception as e:
#             (out_dir / "patient_roc_error.txt").write_text(str(e), encoding="utf-8")


# # =========================
# # 主流程
# # =========================

# def process_one_split(
#     root: Path,
#     split_name: str,
#     out_root: Path,
#     patient_info_df: pd.DataFrame,
#     score_mode: str,
#     threshold: float,
#     min_folds_required: int,
# ):
#     print(f"\n{'=' * 80}")
#     print(f"开始处理 split: {split_name}")
#     print(f"{'=' * 80}")

#     out_dir = out_root / split_name
#     ensure_dir(out_dir)

#     fold_csvs = discover_fold_csvs(root, split_name)
#     if len(fold_csvs) == 0:
#         print(f"⚠️ 未找到 {split_name} 的 fold 结果，跳过")
#         return

#     all_fold_patient_rows = []
#     all_fold_cell_rows = []
#     failed_files = []

#     for csv_path in fold_csvs:
#         fold_id = parse_fold_id(csv_path)
#         print(f"\n读取 fold{fold_id}: {csv_path}")

#         try:
#             df = pd.read_csv(csv_path)
#             patient_df, cell_df = aggregate_one_fold_to_patient_level(df=df, fold_id=fold_id, positive_class=1)

#             if len(patient_df) == 0:
#                 print(f"⚠️ fold{fold_id} 聚合后为空")
#                 continue

#             patient_df.to_csv(out_dir / f"fold{fold_id}_patient_level.csv", index=False, encoding="utf-8-sig")
#             cell_df.to_csv(out_dir / f"fold{fold_id}_cell_level_with_parsed_info.csv", index=False, encoding="utf-8-sig")

#             all_fold_patient_rows.append(patient_df)
#             all_fold_cell_rows.append(cell_df)

#             print(f"✅ fold{fold_id} 患者级聚合完成，共 {len(patient_df)} 位患者")
#         except Exception as e:
#             print(f"❌ fold{fold_id} 处理失败: {e}")
#             failed_files.append({"csv": str(csv_path), "error": str(e)})

#     if len(all_fold_patient_rows) == 0:
#         print(f"⚠️ {split_name} 没有成功处理的 fold，跳过")
#         if len(failed_files) > 0:
#             pd.DataFrame(failed_files).to_csv(out_dir / "failed_files.csv", index=False, encoding="utf-8-sig")
#         return

#     patient_fold_df = pd.concat(all_fold_patient_rows, axis=0, ignore_index=True)
#     patient_fold_df = patient_fold_df.sort_values(["patient_id", "fold_id"]).reset_index(drop=True)
#     patient_fold_df.to_csv(out_dir / "all_fold_patient_level_long.csv", index=False, encoding="utf-8-sig")

#     # ensemble
#     ensemble_df = patient_level_ensemble(
#         patient_fold_df=patient_fold_df,
#         score_mode=score_mode,
#         threshold=threshold,
#         min_folds_required=min_folds_required,
#     )

#     # 合并患者标签
#     if len(patient_info_df) > 0:
#         label_df = patient_info_df[["正式编号", "患者大类型", "true_label"]].copy()
#         label_df = label_df.rename(columns={"正式编号": "patient_id", "患者大类型": "type"})
#         ensemble_df = ensemble_df.merge(label_df, on="patient_id", how="left")

#     # 输出excel
#     excel_path = out_dir / "patient_level_ensemble.xlsx"
#     with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
#         ensemble_df.to_excel(writer, sheet_name="patient_ensemble", index=False)
#         patient_fold_df.to_excel(writer, sheet_name="all_fold_patient_level_long", index=False)

#     ensemble_df.to_csv(out_dir / "patient_level_ensemble.csv", index=False, encoding="utf-8-sig")
#     print(f"✅ 已保存患者级 ensemble 结果: {out_dir / 'patient_level_ensemble.csv'}")

#     # 指标
#     compute_patient_metrics(
#         df=ensemble_df,
#         out_dir=out_dir,
#         threshold=threshold,
#     )

#     # 关键图：参考用户给的 patient_analysis 代码
#     plot_patient_ratios(
#         ensemble_df[["patient_id", "actual_ratio", "predicted_ratio", "type"]].copy(),
#         out_dir / "patient_ratio_compare_from_ensemble.png",
#         title="各患者原始细胞比例对比 (实际 vs 预测, 5-fold ensemble)"
#     )

#     meta = {
#         "split_name": split_name,
#         "n_fold_csvs_found": len(fold_csvs),
#         "n_fold_csvs_success": len(all_fold_patient_rows),
#         "score_mode": score_mode,
#         "threshold": threshold,
#         "min_folds_required": min_folds_required,
#     }
#     (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

#     if len(failed_files) > 0:
#         pd.DataFrame(failed_files).to_csv(out_dir / "failed_files.csv", index=False, encoding="utf-8-sig")

#     print(f"✅ split={split_name} 处理完成")


# def parse_args():
#     p = argparse.ArgumentParser(description="方案二：患者层面 5-fold ensemble，并输出参考 patient_analysis 的比例对比图")
#     p.add_argument("--root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold_260403")
#     p.add_argument("--out-root", default=None)
#     p.add_argument("--splits", nargs="*", default=["test_BJH", "test_FXH_noALL", "test_TJMU"])
#     p.add_argument("--patient-xlsx", default="/root/autodl-tmp/data/patient_data_260323.xlsx")
#     p.add_argument("--sheet-name", default="总表")
#     p.add_argument("--score-mode", default="mean_prob", choices=["mean_prob", "ratio", "vote"])
#     p.add_argument("--threshold", type=float, default=0.5)
#     p.add_argument("--min-folds-required", type=int, default=1)
#     return p.parse_args()


# def main():
#     set_chinese_font()

#     args = parse_args()
#     root = Path(args.root)
#     out_root = Path(args.out_root) if args.out_root else (root / "ensemble_eval_patient_level")
#     ensure_dir(out_root)

#     patient_xlsx = Path(args.patient_xlsx) if args.patient_xlsx else None
#     patient_info_df = load_patient_info(patient_xlsx, sheet_name=args.sheet_name)

#     print(f"root = {root}")
#     print(f"out_root = {out_root}")
#     print(f"patient_xlsx = {patient_xlsx}")
#     print(f"score_mode = {args.score_mode}")
#     print(f"threshold = {args.threshold}")
#     print(f"min_folds_required = {args.min_folds_required}")
#     print(f"患者信息表记录数 = {len(patient_info_df)}")

#     for split_name in args.splits:
#         process_one_split(
#             root=root,
#             split_name=split_name,
#             out_root=out_root,
#             patient_info_df=patient_info_df,
#             score_mode=args.score_mode,
#             threshold=args.threshold,
#             min_folds_required=args.min_folds_required,
#         )

#     print("\n全部处理完成。")


# if __name__ == "__main__":
#     main()

    
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
    p.add_argument("--root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold_260403", help="包含 fold*/eval 的根目录")
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