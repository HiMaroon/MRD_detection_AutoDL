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

SPLIT_GEOMETRY_FEATURES = [
    "area", "perimeter", "convex_area", "equiv_diameter", "major_axis_length", "minor_axis_length",
]
PRE_CROP_GEOMETRY_FEATURES = [f"pre_crop_{k}" for k in SPLIT_GEOMETRY_FEATURES]
POST_CROP_GEOMETRY_FEATURES = [f"post_crop_{k}" for k in SPLIT_GEOMETRY_FEATURES]
EXTRA_GEOMETRY_FEATURES = PRE_CROP_GEOMETRY_FEATURES + POST_CROP_GEOMETRY_FEATURES


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


def resolve_dynamic_crop_size(contour, scale_factor, min_crop_size, max_crop_size):
    _, _, w, h = cv2.boundingRect(contour.astype(np.int32))
    roi_size = max(int(w), int(h))
    dynamic_size = int(round(roi_size * scale_factor))
    dynamic_size = max(min_crop_size, dynamic_size)
    dynamic_size = min(max_crop_size, dynamic_size)
    return dynamic_size


def compute_square_crop(center_x, center_y, crop_size, img_w, img_h):
    x1 = max(0, int(round(center_x - crop_size / 2)))
    y1 = max(0, int(round(center_y - crop_size / 2)))
    x2 = min(img_w, x1 + crop_size)
    y2 = min(img_h, y1 + crop_size)

    if x2 - x1 < crop_size:
        if x1 == 0:
            x2 = min(img_w, crop_size)
        else:
            x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        if y1 == 0:
            y2 = min(img_h, crop_size)
        else:
            y1 = max(0, y2 - crop_size)
    return x1, y1, x2, y2


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


def extract_geometry_subset(prefix, contour_feat):
    return {f"{prefix}_{k}": float(contour_feat.get(k, 0.0)) for k in SPLIT_GEOMETRY_FEATURES}


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

        valid_dets.append({
            "raw_det_idx": i,
            "saved_idx": len(valid_dets),
            "contour_local_base": contour_local,
            "contour_global": contour.copy().astype(np.float32),
        })

    return valid_dets


def empty_feature_dict(feature_list, extra_feature_list=None):
    keys = list(feature_list)
    if extra_feature_list:
        keys.extend(extra_feature_list)
    return {k: 0.0 for k in keys}


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
        r.update(empty_feature_dict(feature_list, EXTRA_GEOMETRY_FEATURES))
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
        contour_local_base = det["contour_local_base"].copy().astype(np.float32)
        contour_local = scale_contour_to_crop(contour_local_base, crop_img.shape[1], crop_img.shape[0], cfg["crop_size"])

        # 原有特征仍然基于真实 576 裁剪图的坐标系，不改现有流程
        real_contour_feat = compute_contour_features(contour_local)
        appear_feat = compute_weak_appearance_features(crop_img, contour_local)

        # 额外几何量：按 yolo2singlecell_dynamic_morph.py 的动态裁剪逻辑做“假裁剪”
        contour_global = det["contour_global"].copy().astype(np.float32)
        box = cv2.boundingRect(contour_global.astype(np.int32))
        x, y, w, h = box
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        fake_crop_size = resolve_dynamic_crop_size(
            contour_global,
            scale_factor=cfg["crop_scale_factor"],
            min_crop_size=cfg["min_crop_size"],
            max_crop_size=cfg["max_crop_size"],
        )
        fake_x1, fake_y1, fake_x2, fake_y2 = compute_square_crop(
            center_x, center_y, fake_crop_size, orig_img.shape[1], orig_img.shape[0]
        )
        fake_contour_local = contour_global.copy().astype(np.float32)
        fake_contour_local[:, :, 0] -= fake_x1
        fake_contour_local[:, :, 1] -= fake_y1
        fake_output_size = cfg["fake_output_size"]
        fake_contour_post = scale_contour_to_crop(
            fake_contour_local, fake_output_size, fake_output_size, fake_crop_size
        )

        pre_crop_contour_feat = compute_contour_features(fake_contour_local)
        post_crop_contour_feat = compute_contour_features(fake_contour_post)

        all_feat = {**real_contour_feat, **appear_feat}
        split_geometry_feat = {
            **extract_geometry_subset("pre_crop", pre_crop_contour_feat),
            **extract_geometry_subset("post_crop", post_crop_contour_feat),
        }
        valid = 1
        reason = "ok"
        if real_contour_feat["area"] < 10 or real_contour_feat["perimeter"] <= 1e-6:
            valid, reason = 0, "invalid_geometry"

        row = {
            "source_type": "yolo", "image_path": item["img_path"], "json_path": str(seg_json_path),
            "filename": Path(item["img_path"]).name, "big_label": item["big_label"], "small_label": item["small_label"],
            "valid": int(valid), "multi_object_flag": int(multi_object_flag), "reason": reason,
            "orig_stem": task["orig_stem"], "obj_idx": obj_idx,
        }
        row.update({k: float(all_feat.get(k, 0.0)) for k in feature_list})
        row.update(split_geometry_feat)
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
        "min_crop_size": args.min_crop_size,
        "max_crop_size": args.max_crop_size,
        "crop_scale_factor": args.crop_scale_factor,
        "fake_output_size": args.fake_output_size,
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

    fixed_cols = [
        "source_type", "image_path", "json_path", "filename", "big_label", "small_label",
        *feature_list,
        *EXTRA_GEOMETRY_FEATURES,
        "valid", "multi_object_flag", "reason", "orig_stem", "obj_idx",
    ]
    df = df[fixed_cols]

    out_csv = out_root / f"{split_cfg['name']}_morph.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Done] csv saved to: {out_csv}")
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
    parser.add_argument("--min_crop_size", type=int, default=100)
    parser.add_argument("--max_crop_size", type=int, default=600)
    parser.add_argument("--crop_scale_factor", type=float, default=1.2)
    parser.add_argument("--fake_output_size", type=int, default=224, help="假裁剪后的目标输出尺寸，仿照 yolo2singlecell_dynamic_morph.py 的 output_size")
    parser.add_argument("--filter_edge_cells", action="store_true")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--qa_max", type=int, default=100)
    parser.add_argument("--print_all_features", action="store_true")
    args = parser.parse_args()

    if args.print_all_features:
        print("Contour:", CONTOUR_FEATURES)
        print("Weak appearance:", WEAK_APPEARANCE_FEATURES)
        print("All:", ALL_FEATURES)
        print("Extra geometry (always exported):", EXTRA_GEOMETRY_FEATURES)
        print("Fake dynamic crop output size:", args.fake_output_size)
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
python tools/build_morph_yolo_fake_dynamic_geom_v2.py \
  --splits_json tools/morph_splits.json \
  --out_root /root/autodl-tmp/projects/myq/SingleCellProject/dataset/morph_csv_fake_dynamic \
  --features contour+weak \
  --crop_size 576 \
  --min_circularity 0.65 \
  --min_area 10000 \
  --min_crop_size 100 \
  --max_crop_size 600 \
  --crop_scale_factor 1.2 \
  --fake_output_size 224 \
  --filter_edge_cells \
  --max_workers 16 \
  --qa_max 100
'''