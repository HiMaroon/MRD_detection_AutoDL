from __future__ import annotations

import argparse
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _col_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in str(cell_ref) if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return max(n - 1, 0)


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS}si"):
                shared.append("".join((t.text or "") for t in si.iter(f"{NS}t")))

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        first = wb.find(f"{NS}sheets").find(f"{NS}sheet")
        rid = first.attrib[f"{RNS}id"]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{PKG}Relationship"):
            if rel.attrib["Id"] == rid:
                target = rel.attrib["Target"]
                break
        if target is None:
            return pd.DataFrame()
        target = target.lstrip("/")
        sheet_path = target if target.startswith("xl/") else "xl/" + target
        root = ET.fromstring(z.read(sheet_path))

        rows = []
        for row in root.iter(f"{NS}row"):
            vals = []
            for c in row.findall(f"{NS}c"):
                idx = _col_idx(c.attrib.get("r", "A1"))
                while len(vals) < idx:
                    vals.append(None)
                typ = c.attrib.get("t")
                v = c.find(f"{NS}v")
                is_el = c.find(f"{NS}is")
                val = None
                if typ == "s" and v is not None:
                    val = shared[int(v.text)]
                elif typ == "inlineStr" and is_el is not None:
                    val = "".join((t.text or "") for t in is_el.iter(f"{NS}t"))
                elif v is not None:
                    txt = v.text
                    try:
                        val = float(txt)
                        if val.is_integer():
                            val = int(val)
                    except Exception:
                        val = txt
                vals.append(val)
            rows.append(vals)
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    header = [str(x) if x is not None else f"col{i}" for i, x in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=header)


def normalize_type(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    if x in {"HC", "HD", "NORMAL", "HEALTHY"}:
        return "HC"
    if x == "AML":
        return "AML"
    return x


def load_patient_info(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = read_xlsx_first_sheet(path)
    else:
        return None
    if "正式编号" not in df.columns or "患者大类型" not in df.columns:
        return None
    out = df[["正式编号", "患者大类型"]].copy()
    out["正式编号"] = out["正式编号"].astype(str).str.strip()
    out["type"] = out["患者大类型"].apply(normalize_type)
    return out[["正式编号", "type"]].drop_duplicates()


def parse_image_info(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(str(image_name)).stem
    prefix = stem.split("_")[0]
    parts = prefix.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}", parts[2]
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}", None
    return None, None


def binary_counts(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def roc_auc_manual(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    thresholds = np.array([np.inf, *sorted(np.unique(score), reverse=True), -np.inf], dtype=float)
    fprs, tprs = [], []
    for thr in thresholds:
        pred = (score >= thr).astype(int)
        tn, fp, fn, tp = binary_counts(y_true, pred)
        fprs.append(fp / max(fp + tn, 1))
        tprs.append(tp / max(tp + fn, 1))
    order = np.argsort(fprs)
    return float(np.trapz(np.asarray(tprs)[order], np.asarray(fprs)[order]))


def classification_soft_metrics(df: pd.DataFrame) -> dict:
    y_true = (df["true_label"].astype(int) > 0).astype(int).to_numpy()
    y_pred = (df["pred_label"].astype(int) > 0).astype(int).to_numpy()
    score = (df.get("prob_class_1", 0).astype(float) + df.get("prob_class_2", 0).astype(float)).to_numpy()
    tn, fp, fn, tp = binary_counts(y_true, y_pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {
        "soft_accuracy_nm": float((y_true == y_pred).mean()),
        "soft_precision_nm": float(precision),
        "soft_sensitivity_nm": float(recall),
        "soft_specificity_other": float(specificity),
        "soft_f1_nm": float(f1),
        "soft_tn": tn,
        "soft_fp": fp,
        "soft_fn": fn,
        "soft_tp": tp,
    }
    auc = roc_auc_manual(y_true, score)
    if auc is not None:
        out["soft_auc_nm"] = auc
    return out


def build_patient_summary(cell_df: pd.DataFrame, patient_info: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = cell_df.copy()
    parsed = df["image"].apply(parse_image_info)
    df["patient_id"] = parsed.apply(lambda x: x[0])
    df["smear_id"] = parsed.apply(lambda x: x[1])
    df = df[df["patient_id"].notna()].copy()
    for c in ["prob_class_0", "prob_class_1", "prob_class_2"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["true_other"] = (df["true_label"].astype(int) == 0).astype(int)
    df["true_N"] = (df["true_label"].astype(int) == 1).astype(int)
    df["true_M"] = (df["true_label"].astype(int) == 2).astype(int)
    df["true_NM"] = (df["true_label"].astype(int) > 0).astype(int)
    df["pred_other"] = (df["pred_label"].astype(int) == 0).astype(int)
    df["pred_N"] = (df["pred_label"].astype(int) == 1).astype(int)
    df["pred_M"] = (df["pred_label"].astype(int) == 2).astype(int)
    df["pred_NM"] = (df["pred_label"].astype(int) > 0).astype(int)
    df["prob_NM"] = df["prob_class_1"] + df["prob_class_2"]
    df["strict_correct"] = (df["true_label"].astype(int) == df["pred_label"].astype(int)).astype(int)
    df["soft_correct"] = (df["true_NM"] == df["pred_NM"]).astype(int)

    summary = df.groupby("patient_id").agg(
        n_cells=("image", "count"),
        n_smears=("smear_id", "nunique"),
        actual_N_ratio=("true_N", "mean"),
        actual_M_ratio=("true_M", "mean"),
        actual_NM_ratio=("true_NM", "mean"),
        pred_N_ratio_hard=("pred_N", "mean"),
        pred_M_ratio_hard=("pred_M", "mean"),
        pred_NM_ratio_hard=("pred_NM", "mean"),
        mean_prob_other=("prob_class_0", "mean"),
        mean_prob_N=("prob_class_1", "mean"),
        mean_prob_M=("prob_class_2", "mean"),
        mean_prob_NM=("prob_NM", "mean"),
        p90_prob_NM=("prob_NM", lambda x: float(np.quantile(x, 0.90))),
        p95_prob_NM=("prob_NM", lambda x: float(np.quantile(x, 0.95))),
        strict_accuracy_3class=("strict_correct", "mean"),
        soft_accuracy_NM=("soft_correct", "mean"),
    ).reset_index()
    summary["max_mean_prob_N_or_M"] = summary[["mean_prob_N", "mean_prob_M"]].max(axis=1)
    summary["max_pred_ratio_N_or_M"] = summary[["pred_N_ratio_hard", "pred_M_ratio_hard"]].max(axis=1)

    if patient_info is not None:
        summary = summary.merge(patient_info, left_on="patient_id", right_on="正式编号", how="left")
        summary = summary.drop(columns=["正式编号"])
    else:
        summary["type"] = np.nan
    return summary


def patient_threshold_metrics(summary: pd.DataFrame, score_col: str, thresholds) -> pd.DataFrame:
    df = summary.copy()
    df = df[df["type"].isin(["HC", "AML"])].copy()
    if len(df) == 0:
        return pd.DataFrame()
    y_true = (df["type"] == "AML").astype(int).to_numpy()
    rows = []
    for thr in thresholds:
        y_pred = (df[score_col].astype(float).to_numpy() >= thr).astype(int)
        tn, fp, fn, tp = binary_counts(y_true, y_pred)
        rows.append({
            "score_col": score_col,
            "threshold": float(thr),
            "n_patients": int(len(df)),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": float((y_true == y_pred).mean()),
            "specificity_hc": float(tn / max(tn + fp, 1)),
            "sensitivity_aml": float(tp / max(tp + fn, 1)),
            "youden": float(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1),
        })
    out = pd.DataFrame(rows)
    auc = roc_auc_manual(y_true, df[score_col].astype(float).to_numpy())
    if auc is not None:
        out["patient_auc"] = auc
    return out


def process_split(results_csv: Path, out_dir: Path, patient_info: Optional[pd.DataFrame]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    metrics = classification_soft_metrics(df)
    summary = build_patient_summary(df, patient_info)
    thresholds = np.round(np.arange(0.0, 1.0001, 0.05), 6)
    sweeps = []
    for score_col in ["mean_prob_NM", "max_mean_prob_N_or_M", "pred_NM_ratio_hard", "max_pred_ratio_N_or_M"]:
        sweeps.append(patient_threshold_metrics(summary, score_col, thresholds))
    sweep = pd.concat([x for x in sweeps if len(x) > 0], ignore_index=True) if sweeps else pd.DataFrame()

    df.to_csv(out_dir / "cell_results_threeclass.csv", index=False)
    summary.to_csv(out_dir / "patient_summary_threeclass_nm.csv", index=False)
    if len(sweep) > 0:
        sweep.to_csv(out_dir / "patient_threshold_sweep_threeclass_nm.csv", index=False)
    with open(out_dir / "soft_cell_metrics_nm.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return {"split": out_dir.name, **metrics, "n_patients": int(len(summary))}


def main():
    ap = argparse.ArgumentParser(description="Three-class N/M evaluation with soft NM screening patient metrics.")
    ap.add_argument("--results-root", type=Path, required=True, help="Root containing split/val_results.csv files.")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--patient-info-xlsx", type=Path, default=Path("/root/autodl-tmp/data/patient_data_260416.xlsx"))
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test_BJH", "test_FXH_noALL", "test_TJMU"])
    args = ap.parse_args()
    patient_info = load_patient_info(args.patient_info_xlsx)
    rows = []
    for split in args.splits:
        csv_path = args.results_root / split / "val_results.csv"
        if not csv_path.exists():
            print(f"skip missing {csv_path}")
            continue
        rows.append(process_split(csv_path, args.out_root / split, patient_info))
    if rows:
        pd.DataFrame(rows).to_csv(args.out_root / "all_splits_soft_cell_metrics_nm.csv", index=False)
        print(f"saved {args.out_root}")


if __name__ == "__main__":
    main()
