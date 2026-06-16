import argparse
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

COMMON_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

CONTOUR_FEATURES = [
    "area", "perimeter", "circularity", "aspect_ratio", "extent",
    "convex_area", "solidity", "equiv_diameter", "major_axis_length",
    "minor_axis_length", "eccentricity",
]

WEAK_APPEARANCE_FEATURES = [
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    "gray_mean", "gray_std", "entropy", "texture_energy", "texture_contrast",
    "laplacian_var", "dark_region_ratio", "low_saturation_ratio", "central_compactness",
]

ALL_FEATURES = CONTOUR_FEATURES + WEAK_APPEARANCE_FEATURES


# ===== 针对 MLP 的特征分组预处理 =====
# 思路：
# 1) 尺度/长度类通常右偏，先做 log 压缩，再做 z-score；
# 2) 比例/概率类本身有明确范围，先 clip 到合法范围，再做 z-score；
# 3) RGB/HSV 等有固定物理量程，先除以量程归一化，再做 z-score；
# 4) 熵有明确上界（8bit 灰度最大约 8），先除以 8，再做 z-score。
LOG1P_ZSCORE_FEATURES = {
    "area", "perimeter", "convex_area", "equiv_diameter",
    "major_axis_length", "minor_axis_length", "laplacian_var",
}

LOG_ZSCORE_FEATURES = {
    "aspect_ratio",
}

UNIT_INTERVAL_CLIP_ZSCORE_FEATURES = {
    "circularity", "extent", "solidity", "eccentricity",
    "gray_mean", "gray_std", "entropy", "texture_energy",
    "texture_contrast", "dark_region_ratio", "low_saturation_ratio",
    "central_compactness",
}

FIXED_RANGE_255_ZSCORE_FEATURES = {
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_s", "mean_v", "std_s", "std_v",
}

FIXED_RANGE_179_ZSCORE_FEATURES = {
    "mean_h", "std_h",
}

SPECIAL_MAX_FEATURES = {
    "entropy": 8.0,  # 8-bit 灰度熵理论上限约为 8
}

DEFAULT_ZSCORE_FEATURES = set()


def strip_feature_prefix(feature_name: str) -> str:
    for prefix in ("pre_crop_", "post_crop_", "norm_"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name


def get_feature_preprocess_config(feature_name: str) -> dict:
    base = strip_feature_prefix(feature_name)

    if base in LOG1P_ZSCORE_FEATURES:
        return {"type": "log1p_zscore"}
    if base in LOG_ZSCORE_FEATURES:
        return {"type": "log_zscore"}
    if base in FIXED_RANGE_255_ZSCORE_FEATURES:
        return {"type": "fixed_range_zscore", "range_max": 255.0}
    if base in FIXED_RANGE_179_ZSCORE_FEATURES:
        return {"type": "fixed_range_zscore", "range_max": 179.0}
    if base in SPECIAL_MAX_FEATURES:
        return {"type": "fixed_range_zscore", "range_max": float(SPECIAL_MAX_FEATURES[base])}
    if base in UNIT_INTERVAL_CLIP_ZSCORE_FEATURES:
        return {"type": "clip01_zscore"}

    return {"type": "zscore"}


def preprocess_feature_values(values: np.ndarray, feature_name: str) -> tuple[np.ndarray, dict]:
    cfg = get_feature_preprocess_config(feature_name)
    method = cfg["type"]
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)

    if method == "log1p_zscore":
        x = np.maximum(x, 0.0)
        x = np.log1p(x)
    elif method == "log_zscore":
        x = np.log(np.clip(x, 1e-6, None))
    elif method == "fixed_range_zscore":
        range_max = float(cfg["range_max"])
        x = np.clip(x / range_max, 0.0, 1.0)
    elif method == "clip01_zscore":
        x = np.clip(x, 0.0, 1.0)
    else:
        # 普通 z-score，不做前置非线性变换
        pass

    return x, cfg



def read_label_txt(txt_path):
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.rsplit(maxsplit=2)
            if len(parts) != 3:
                continue
            samples.append((parts[0], int(parts[1]), int(parts[2])))
    return samples


def parse_singlecell_name(img_path):
    stem = Path(img_path).stem
    m = re.match(r"^(.*)_(\d+)_([A-Za-z0-9]+)$", stem)
    if m is None:
        raise ValueError(f"无法解析单细胞文件名: {stem}")
    return m.group(1), int(m.group(2)), m.group(3)


def infer_patient_from_orig_stem(orig_stem: str):
    parts = orig_stem.split("-")
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else parts[0]


def resolve_image_path(orig_root, patient, orig_stem):
    patient_dir = Path(orig_root) / patient
    for ext in COMMON_IMAGE_EXTS:
        p = patient_dir / f"{orig_stem}{ext}"
        if p.exists():
            return str(p)
    for ext in COMMON_IMAGE_EXTS:
        found = list(Path(orig_root).rglob(f"{orig_stem}{ext}"))
        if found:
            return str(found[0])
    return None


def resolve_seg_json_path(seg_root, patient, orig_stem):
    p = Path(seg_root) / patient / f"{orig_stem}.json"
    if p.exists():
        return str(p)
    found = list(Path(seg_root).rglob(f"{orig_stem}.json"))
    return str(found[0]) if found else None


def get_contour_from_segments(cell_data):
    seg = cell_data.get("segments", {})
    if "x" in seg and "y" in seg:
        pts = np.array([[int(x), int(y)] for x, y in zip(seg["x"], seg["y"])], dtype=np.int32)
        if len(pts) >= 3:
            return pts.reshape(-1, 1, 2)
    return None


def is_cell_complete(contour, crop_size, margin=10):
    x, y, w, h = cv2.boundingRect(contour)
    return (x > margin and y > margin and (x + w) < (crop_size - margin) and (y + h) < (crop_size - margin))


def scale_contour_to_crop(contour_base, crop_w, crop_h, crop_size):
    contour = contour_base.copy().astype(np.float32)
    contour[:, :, 0] *= crop_w / float(crop_size)
    contour[:, :, 1] *= crop_h / float(crop_size)
    contour[:, :, 0] = np.clip(contour[:, :, 0], 0, max(crop_w - 1, 0))
    contour[:, :, 1] = np.clip(contour[:, :, 1], 0, max(crop_h - 1, 0))
    return contour


def compute_contour_features(contour):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0

    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    aspect_ratio = float(w / h) if h > 0 else 0.0
    extent = float(area / (w * h)) if (w > 0 and h > 0) else 0.0

    hull = cv2.convexHull(contour.astype(np.float32))
    convex_area = float(cv2.contourArea(hull))
    solidity = float(area / convex_area) if convex_area > 1e-6 else 0.0

    equiv_diameter = float(math.sqrt(4.0 * area / math.pi)) if area > 0 else 0.0
    major_axis_length, minor_axis_length, eccentricity = 0.0, 0.0, 0.0

    if len(contour) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(contour.astype(np.float32))
        major_axis_length, minor_axis_length = float(max(ma, mi)), float(min(ma, mi))
        if major_axis_length > 1e-6:
            eccentricity = float(math.sqrt(max(0.0, 1.0 - (minor_axis_length ** 2) / (major_axis_length ** 2))))

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "convex_area": convex_area,
        "solidity": solidity,
        "equiv_diameter": equiv_diameter,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "eccentricity": eccentricity,
    }


def masked_stats(arr, mask):
    vals = arr[mask > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())


def entropy_8bit(gray_u8, mask):
    vals = gray_u8[mask > 0]
    if vals.size == 0:
        return 0.0
    hist = cv2.calcHist([vals.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def glcm_like_texture(gray_f32, mask):
    # 轻量替代 GLCM：使用灰度差分统计
    g = gray_f32
    m = mask > 0
    if g.shape[1] < 2:
        return 0.0, 0.0
    diff = np.abs(g[:, 1:] - g[:, :-1])
    m2 = m[:, 1:] & m[:, :-1]
    v = diff[m2]
    if v.size == 0:
        return 0.0, 0.0
    energy = float(np.mean((1.0 - np.clip(v, 0, 1)) ** 2))
    contrast = float(np.mean(v ** 2))
    return energy, contrast


def compute_weak_appearance_features(crop_bgr, contour):
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, thickness=-1)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    mean_r, std_r = masked_stats(rgb[:, :, 0], mask)
    mean_g, std_g = masked_stats(rgb[:, :, 1], mask)
    mean_b, std_b = masked_stats(rgb[:, :, 2], mask)
    mean_h, std_h = masked_stats(hsv[:, :, 0], mask)
    mean_s, std_s = masked_stats(hsv[:, :, 1], mask)
    mean_v, std_v = masked_stats(hsv[:, :, 2], mask)
    gray_mean, gray_std = masked_stats(gray, mask)

    ent = entropy_8bit(gray_u8, mask)
    texture_energy, texture_contrast = glcm_like_texture(gray, mask)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_vals = lap[mask > 0]
    laplacian_var = float(np.var(lap_vals)) if lap_vals.size > 0 else 0.0

    dark_region_ratio = float(np.mean((gray < 0.35)[mask > 0])) if np.any(mask > 0) else 0.0
    low_saturation_ratio = float(np.mean((hsv[:, :, 1] < 50)[mask > 0])) if np.any(mask > 0) else 0.0

    ys, xs = np.where(mask > 0)
    if xs.size > 0:
        cx, cy = float(xs.mean()), float(ys.mean())
        yy, xx = np.indices((h, w))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = np.percentile(dist[mask > 0], 40)
        central = ((dist <= r) & (mask > 0))
        central_compactness = float(central.sum() / max((mask > 0).sum(), 1))
    else:
        central_compactness = 0.0

    return {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "std_r": std_r, "std_g": std_g, "std_b": std_b,
        "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v,
        "std_h": std_h, "std_s": std_s, "std_v": std_v,
        "gray_mean": gray_mean, "gray_std": gray_std,
        "entropy": ent, "texture_energy": texture_energy, "texture_contrast": texture_contrast,
        "laplacian_var": laplacian_var,
        "dark_region_ratio": dark_region_ratio,
        "low_saturation_ratio": low_saturation_ratio,
        "central_compactness": central_compactness,
    }


def build_valid_detections(seg_data, orig_w, orig_h, crop_size, min_circularity, min_area, filter_edge_cells):
    valid_dets = []
    for i, cell_data in enumerate(seg_data):
        box = cell_data.get("box", {})
        x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        crop_x1 = max(0, int(cx - crop_size / 2))
        crop_y1 = max(0, int(cy - crop_size / 2))
        crop_x2 = min(orig_w, crop_x1 + crop_size)
        crop_y2 = min(orig_h, crop_y1 + crop_size)
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)

        contour = get_contour_from_segments(cell_data)
        if contour is None:
            continue

        contour_local = contour.copy().astype(np.float32)
        contour_local[:, :, 0] -= crop_x1
        contour_local[:, :, 1] -= crop_y1

        if filter_edge_cells and (not is_cell_complete(contour_local, crop_size)):
            continue

        feat = compute_contour_features(contour_local)
        if feat["area"] < min_area or feat["circularity"] < min_circularity:
            continue

        valid_dets.append({"raw_det_idx": i, "saved_idx": len(valid_dets), "contour_local_base": contour_local})

    return valid_dets


def empty_feature_dict(feature_list):
    return {k: 0.0 for k in feature_list}


def process_group_task(task):
    cfg = task["cfg"]
    seg_json_path, orig_image_path = task["seg_json_path"], task["orig_image_path"]
    rows = []

    feature_list = cfg["feature_list"]

    def add_fail(item, reason, multi=0):
        r = {
            "source_type": "yolo", "image_path": item["img_path"], "json_path": str(seg_json_path) if seg_json_path else "",
            "filename": Path(item["img_path"]).name, "big_label": item["big_label"], "small_label": item["small_label"],
            "valid": 0, "multi_object_flag": int(multi), "reason": reason, "orig_stem": task["orig_stem"], "obj_idx": item["obj_idx"],
        }
        r.update(empty_feature_dict(feature_list))
        rows.append(r)

    if seg_json_path is None or (not os.path.exists(seg_json_path)):
        for item in task["sample_items"]:
            add_fail(item, "seg_json_not_found")
        return rows

    if orig_image_path is None or (not os.path.exists(orig_image_path)):
        for item in task["sample_items"]:
            add_fail(item, "orig_image_not_found")
        return rows

    try:
        with open(seg_json_path, "r", encoding="utf-8") as f:
            seg_data = json.load(f)
    except Exception as e:
        for item in task["sample_items"]:
            add_fail(item, f"seg_json_load_failed:{e}")
        return rows

    orig_img = cv2.imread(orig_image_path)
    if orig_img is None:
        for item in task["sample_items"]:
            add_fail(item, "orig_image_read_failed")
        return rows

    valid_dets = build_valid_detections(
        seg_data=seg_data,
        orig_w=orig_img.shape[1],
        orig_h=orig_img.shape[0],
        crop_size=cfg["crop_size"],
        min_circularity=cfg["min_circularity"],
        min_area=cfg["min_area"],
        filter_edge_cells=cfg["filter_edge_cells"],
    )
    multi_object_flag = 1 if len(seg_data) > 1 else 0

    for item in task["sample_items"]:
        obj_idx = item["obj_idx"]
        if obj_idx < 0 or obj_idx >= len(valid_dets):
            add_fail(item, "saved_index_out_of_range", multi=multi_object_flag)
            continue

        crop_img = cv2.imread(item["img_path"])
        if crop_img is None:
            add_fail(item, "crop_read_failed", multi=multi_object_flag)
            continue

        det = valid_dets[obj_idx]
        contour_local = scale_contour_to_crop(det["contour_local_base"], crop_img.shape[1], crop_img.shape[0], cfg["crop_size"])

        contour_feat = compute_contour_features(contour_local)
        appear_feat = compute_weak_appearance_features(crop_img, contour_local)

        all_feat = {**contour_feat, **appear_feat}
        valid = 1
        reason = "ok"
        if contour_feat["area"] < 10 or contour_feat["perimeter"] <= 1e-6:
            valid, reason = 0, "invalid_geometry"

        row = {
            "source_type": "yolo", "image_path": item["img_path"], "json_path": str(seg_json_path),
            "filename": Path(item["img_path"]).name, "big_label": item["big_label"], "small_label": item["small_label"],
            "valid": int(valid), "multi_object_flag": int(multi_object_flag), "reason": reason,
            "orig_stem": task["orig_stem"], "obj_idx": obj_idx,
        }
        row.update({k: float(all_feat.get(k, 0.0)) for k in feature_list})
        rows.append(row)

    return rows


def build_group_tasks(split_cfg, worker_cfg):
    samples = read_label_txt(split_cfg["label_txt"])
    grouped = defaultdict(list)

    for img_path, big_label, small_label in samples:
        try:
            orig_stem, obj_idx, _ = parse_singlecell_name(img_path)
            patient = infer_patient_from_orig_stem(orig_stem)
            grouped[(patient, orig_stem)].append({"img_path": img_path, "big_label": big_label, "small_label": small_label, "obj_idx": obj_idx})
        except Exception:
            grouped[("__PARSE_FAILED__", img_path)].append({"img_path": img_path, "big_label": big_label, "small_label": small_label, "obj_idx": -1})

    tasks = []
    for (patient, orig_stem), sample_items in grouped.items():
        if patient == "__PARSE_FAILED__":
            tasks.append({"patient": "", "orig_stem": "", "seg_json_path": None, "orig_image_path": None, "sample_items": sample_items, "cfg": worker_cfg})
            continue
        tasks.append({
            "patient": patient,
            "orig_stem": orig_stem,
            "seg_json_path": resolve_seg_json_path(split_cfg["seg_root"], patient, orig_stem),
            "orig_image_path": resolve_image_path(split_cfg["orig_root"], patient, orig_stem),
            "sample_items": sample_items,
            "cfg": worker_cfg,
        })
    return tasks


def save_qa_image(img_bgr, contour, out_path, text_lines):
    canvas = img_bgr.copy()
    cnt = contour.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(canvas, [cnt], True, (0, 255, 0), 2)
    y0 = 22
    for t in text_lines:
        cv2.putText(canvas, t, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        y0 += 20
    cv2.imwrite(str(out_path), canvas)



def build_norm_stats(df, feature_list, norm_method="featurewise"):
    valid_mask = (df["valid"] == 1) & (df["reason"] == "ok")
    if int(valid_mask.sum()) == 0:
        valid_mask = pd.Series([True] * len(df), index=df.index)

    stats = {
        "norm_method": norm_method,
        "feature_order": list(feature_list),
        "features": {},
    }

    for feat in feature_list:
        vals = pd.to_numeric(df.loc[valid_mask, feat], errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            vals = pd.Series([0.0], dtype=float)

        transformed_vals, preprocess_cfg = preprocess_feature_values(vals.to_numpy(dtype=float), feat)
        transformed_vals = pd.Series(transformed_vals).replace([np.inf, -np.inf], np.nan).dropna()
        if len(transformed_vals) == 0:
            transformed_vals = pd.Series([0.0], dtype=float)

        mean = float(transformed_vals.mean())
        std = float(transformed_vals.std(ddof=0))
        if abs(std) < 1e-12:
            std = 1.0

        stats["features"][feat] = {
            "preprocess": preprocess_cfg,
            "mean": mean,
            "std": std,
        }

    return stats


def apply_norm_columns(df, feature_list, stats):
    arr = df[feature_list].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    norm_cols = {}
    for feat in feature_list:
        transformed, _ = preprocess_feature_values(arr[feat].to_numpy(dtype=float), feat)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        s = stats["features"][feat]
        norm_name = f"norm_{feat}"
        norm_cols[norm_name] = (transformed - float(s["mean"])) / float(s["std"])

    norm_df = pd.DataFrame(norm_cols, index=df.index)
    for col in norm_df.columns:
        df[col] = norm_df[col].astype(float)
    return df


def process_split(split_cfg, args, feature_list):

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    qa_dir = out_root / f"qa_{split_cfg['name']}"
    qa_dir.mkdir(parents=True, exist_ok=True)

    worker_cfg = {
        "crop_size": args.crop_size,
        "min_circularity": args.min_circularity,
        "min_area": args.min_area,
        "filter_edge_cells": args.filter_edge_cells,
        "feature_list": feature_list,
    }

    tasks = build_group_tasks(split_cfg, worker_cfg)
    print(f"\\n[Split={split_cfg['name']}] groups={len(tasks)} workers={args.max_workers}")

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(process_group_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_cfg['name']}", ncols=100):
            all_rows.extend(fut.result())

    all_rows = sorted(all_rows, key=lambda x: x["image_path"])
    df = pd.DataFrame(all_rows)

    norm_stats = build_norm_stats(df, feature_list, norm_method=args.norm_method)
    df = apply_norm_columns(df, feature_list, norm_stats)

    norm_feature_cols = [f"norm_{feat}" for feat in feature_list]
    fixed_cols = [
        "source_type", "image_path", "json_path", "filename", "big_label", "small_label",
        *feature_list,
        *norm_feature_cols,
        "valid", "multi_object_flag", "reason", "orig_stem", "obj_idx",
    ]
    df = df[fixed_cols]

    out_csv = out_root / f"{split_cfg['name']}_morph.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    stats_path = out_root / f"{split_cfg['name']}_morph_norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    print(f"[Done] csv saved to: {out_csv}")
    print(f"[Done] norm stats saved to: {stats_path}")
    print(df["reason"].value_counts(dropna=False).head(10))

    qa = df[df["reason"] == "ok"].head(args.qa_max)
    for i, row in enumerate(tqdm(qa.to_dict("records"), desc=f"QA {split_cfg['name']}", ncols=100)):
        img = cv2.imread(row["image_path"])
        if img is None:
            continue
        # QA contour from json redrive for robustness omitted; use simple rectangle if unavailable
        h, w = img.shape[:2]
        dummy = np.array([[[1, 1]], [[w - 2, 1]], [[w - 2, h - 2]], [[1, h - 2]]], dtype=np.int32)
        text = [
            f"valid={int(row['valid'])} multi={int(row['multi_object_flag'])}",
            f"area={row.get('area', 0):.1f} circ={row.get('circularity', 0):.4f}",
            f"entropy={row.get('entropy', 0):.3f} contrast={row.get('texture_contrast', 0):.3f}",
        ]
        save_qa_image(img, dummy, qa_dir / f"{i:04d}_{Path(row['filename']).stem}.png", text)


def parse_features(features_arg):
    if features_arg.lower() in {"all", "*"}:
        return list(ALL_FEATURES)
    if features_arg.lower() == "contour":
        return list(CONTOUR_FEATURES)
    if features_arg.lower() == "contour+weak":
        return list(ALL_FEATURES)
    cols = [c.strip() for c in features_arg.split(",") if c.strip()]
    invalid = [c for c in cols if c not in ALL_FEATURES]
    if invalid:
        raise ValueError(f"Unknown features: {invalid}. Available: {ALL_FEATURES}")
    return cols


def main():
    parser = argparse.ArgumentParser(description="根据 YOLO 轮廓生成 morph csv（支持可选特征列表）")
    parser.add_argument("--splits_json", type=str, required=True, help="json 文件，列表元素含 name/label_txt/seg_root/orig_root")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--features", type=str, default="contour", help="contour | contour+weak | all | col1,col2,...")
    parser.add_argument("--crop_size", type=int, default=576)
    parser.add_argument("--min_circularity", type=float, default=0.65)
    parser.add_argument("--min_area", type=float, default=10000)
    parser.add_argument("--filter_edge_cells", action="store_true")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--qa_max", type=int, default=100)
    parser.add_argument("--print_all_features", action="store_true")
    parser.add_argument("--norm_method", type=str, default="featurewise", choices=["featurewise"], help="按特征类型自动选择预处理后再做 z-score，适合 MLP 输入")
    args = parser.parse_args()

    if args.print_all_features:
        print("Contour:", CONTOUR_FEATURES)
        print("Weak appearance:", WEAK_APPEARANCE_FEATURES)
        print("All:", ALL_FEATURES)
        print("Preprocess groups:")
        for feat in ALL_FEATURES:
            print(f"  {feat}: {get_feature_preprocess_config(feat)}")
        return

    with open(args.splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)

    feature_list = parse_features(args.features)
    print(f"Using features ({len(feature_list)}): {feature_list}")
    for split_cfg in splits:
        process_split(split_cfg, args, feature_list)


if __name__ == "__main__":
    main()

'''
python tools/build_morph_yolo_with_featurewise_norm.py \
  --splits_json tools/morph_splits_trainval.json \
  --out_root dataset/morph_csv_norm_260615 \
  --features contour+weak \
  --crop_size 576 \
  --min_circularity 0.65 \
  --min_area 10000 \
  --filter_edge_cells \
  --max_workers 16 \
  --qa_max 100
'''