import os
import re
import csv
import json
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# =========================================================
# 全局配置
# =========================================================

OUT_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/morph_csv_260414"

CROP_SIZE = 576
MIN_CIRCULARITY = 0.65
MIN_AREA = 10000
FILTER_EDGE_CELLS = True

MAX_WORKERS = 16
QA_MAX_TRAIN = 200
QA_MAX_VAL = 100

COMMON_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

SPLITS = [
    {
        "name": "train_groundtruth",
        "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train_groundtruth_labels.txt",
        "singlecell_root": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train_groundtruth",
        "orig_root": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Train",
        "out_csv": f"{OUT_ROOT}/train_groundtruth_morph.csv",
        "qa_dir": f"{OUT_ROOT}/qa_train_groundtruth_3feat",
        "qa_max": QA_MAX_TRAIN,
    },
    {
        "name": "val_groundtruth",
        "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val_groundtruth_labels.txt",
        "singlecell_root": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val_groundtruth",
        "orig_root": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Val",
        "out_csv": f"{OUT_ROOT}/val_groundtruth_morph.csv",
        "qa_dir": f"{OUT_ROOT}/qa_val_groundtruth_3feat",
        "qa_max": QA_MAX_VAL,
    },
]


# =========================================================
# 基础工具
# =========================================================

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
            img_path = parts[0]
            big_label = int(parts[1])
            small_label = int(parts[2])
            samples.append((img_path, big_label, small_label))
    return samples


def parse_singlecell_name(img_path):
    """
    例:
    BEPH-1-35_000_P2.png
    -> orig_stem = BEPH-1-35
    -> obj_idx = 0
    -> suffix = P2
    """
    stem = Path(img_path).stem
    m = re.match(r"^(.*)_(\d+)_([A-Za-z0-9]+)$", stem)
    if m is None:
        raise ValueError(f"无法解析单细胞文件名: {stem}")
    orig_stem = m.group(1)
    obj_idx = int(m.group(2))   # 0-based, 但这是 valid_cell_count 下标
    suffix = m.group(3)
    return orig_stem, obj_idx, suffix


def load_labelme_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth_polygons(points_json_path):
    """
    对齐 groundtruth2singlecell.py：
    返回原始 polygon 列表，顺序与 json shapes 一致
    """
    gt_polygons = []
    data = load_labelme_json(points_json_path)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") not in ["polygon", "polyline"]:
            continue
        if not shape.get("points"):
            continue

        pts = shape["points"]
        contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        label = shape.get("label", "0")

        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        gt_polygons.append({
            "contour": contour,
            "label": label,
            "points": pts,
            "center_x": center_x,
            "center_y": center_y,
            "bbox": cv2.boundingRect(contour),  # (x, y, w, h)
        })

    image_w = data.get("imageWidth", None)
    image_h = data.get("imageHeight", None)
    return gt_polygons, image_w, image_h


def resolve_image_path(orig_root, patient, orig_stem):
    patient_dir = Path(orig_root) / patient

    for ext in COMMON_IMAGE_EXTS:
        p = patient_dir / f"{orig_stem}{ext}"
        if p.exists():
            return str(p)

    for ext in COMMON_IMAGE_EXTS:
        found = list(Path(orig_root).rglob(f"{orig_stem}{ext}"))
        if len(found) > 0:
            return str(found[0])

    return None


def is_cell_complete_by_orig_bbox(gt_data, orig_img_shape, margin=10):
    """
    严格对齐 groundtruth2singlecell.py 里的边缘过滤：
    它实际检查的是原图上的 bbox 是否贴边
    """
    orig_x, orig_y, orig_w, orig_h = gt_data["bbox"]
    img_h, img_w = orig_img_shape

    if (
        orig_x <= margin
        or orig_x + orig_w >= img_w - margin
        or orig_y <= margin
        or orig_y + orig_h >= img_h - margin
    ):
        return False
    return True


def build_local_contour_for_center_crop(raw_contour, center_x, center_y, crop_size, orig_w, orig_h):
    """
    严格复现 groundtruth2singlecell.py 的裁剪逻辑
    """
    crop_x1_img = max(0, int(center_x - crop_size / 2))
    crop_y1_img = max(0, int(center_y - crop_size / 2))
    crop_x2_img = min(orig_w, crop_x1_img + crop_size)
    crop_y2_img = min(orig_h, crop_y1_img + crop_size)

    if crop_x2_img - crop_x1_img < crop_size:
        if crop_x1_img == 0:
            crop_x2_img = crop_size
        else:
            crop_x1_img = crop_x2_img - crop_size

    if crop_y2_img - crop_y1_img < crop_size:
        if crop_y1_img == 0:
            crop_y2_img = crop_size
        else:
            crop_y1_img = crop_y2_img - crop_size

    contour_local = raw_contour.copy().astype(np.float32)
    contour_local[:, :, 0] -= crop_x1_img
    contour_local[:, :, 1] -= crop_y1_img

    return contour_local, crop_x1_img, crop_y1_img, crop_x2_img, crop_y2_img


def scale_contour_to_crop(contour_base, crop_w, crop_h, crop_size):
    contour = contour_base.copy().astype(np.float32)
    sx = crop_w / float(crop_size)
    sy = crop_h / float(crop_size)
    contour[:, :, 0] *= sx
    contour[:, :, 1] *= sy
    return contour


def polygon_area(poly):
    return float(cv2.contourArea(poly.astype(np.float32)))


def polygon_perimeter(poly):
    return float(cv2.arcLength(poly.astype(np.float32), True))


def polygon_circularity(area, perimeter):
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def save_qa_image(img_bgr, contour, out_path, text_lines):
    canvas = img_bgr.copy()
    cnt = contour.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(canvas, [cnt], isClosed=True, color=(0, 255, 0), thickness=2)

    y0 = 22
    for t in text_lines:
        cv2.putText(canvas, t, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        y0 += 22

    cv2.imwrite(str(out_path), canvas)


# =========================================================
# 关键：复现 groundtruth2singlecell.py 的 valid_cell_count 顺序
# =========================================================

def build_valid_gt_detections(gt_polygons, orig_h, orig_w,
                              crop_size=576,
                              min_circularity=0.65,
                              min_area=10000,
                              filter_edge_cells=True):
    """
    这里的顺序必须和 groundtruth2singlecell.py 最终保存单细胞图的顺序一致。
    文件名里的 000/001/002 对应的是这里 valid_polys 的顺序，不是原始 json 下标。
    """
    valid_polys = []

    for i, gt_data in enumerate(gt_polygons):
        center_x = gt_data["center_x"]
        center_y = gt_data["center_y"]
        raw_contour = gt_data["contour"]

        contour_local, crop_x1, crop_y1, crop_x2, crop_y2 = build_local_contour_for_center_crop(
            raw_contour=raw_contour,
            center_x=center_x,
            center_y=center_y,
            crop_size=crop_size,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        if filter_edge_cells:
            if not is_cell_complete_by_orig_bbox(gt_data, (orig_h, orig_w), margin=10):
                continue

        area = cv2.contourArea(contour_local)
        perimeter = cv2.arcLength(contour_local, True)

        if area < min_area:
            continue

        if perimeter == 0:
            circularity = 0.0
        else:
            circularity = (4 * math.pi * area) / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        valid_polys.append({
            "raw_idx": i,
            "saved_idx": len(valid_polys),  # 这才对应单细胞文件名里的 000/001/002
            "label": gt_data["label"],
            "contour_local_base": contour_local.astype(np.float32),
            "area_base": float(area),
            "perimeter_base": float(perimeter),
            "circularity_base": float(circularity),
        })

    return valid_polys


# =========================================================
# 并行任务
# =========================================================

def process_group_task(task):
    """
    一个 task 对应一个原图（一个 json）下的多个单细胞样本
    """
    patient = task["patient"]
    orig_stem = task["orig_stem"]
    json_path = task["json_path"]
    orig_image_path = task["orig_image_path"]
    sample_items = task["sample_items"]

    rows = []

    if json_path is None or (not os.path.exists(json_path)):
        for item in sample_items:
            rows.append({
                "source_type": "gt",
                "image_path": item["img_path"],
                "json_path": str(json_path) if json_path else "",
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": "json_not_found",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    if orig_image_path is None or (not os.path.exists(orig_image_path)):
        for item in sample_items:
            rows.append({
                "source_type": "gt",
                "image_path": item["img_path"],
                "json_path": str(json_path),
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": "orig_image_not_found",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    try:
        gt_polygons, image_w, image_h = load_ground_truth_polygons(json_path)
    except Exception as e:
        for item in sample_items:
            rows.append({
                "source_type": "gt",
                "image_path": item["img_path"],
                "json_path": str(json_path),
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": f"json_load_failed:{e}",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    if image_w is None or image_h is None:
        for item in sample_items:
            rows.append({
                "source_type": "gt",
                "image_path": item["img_path"],
                "json_path": str(json_path),
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": len(gt_polygons) > 1,
                "reason": "missing_image_size_in_json",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    valid_polys = build_valid_gt_detections(
        gt_polygons=gt_polygons,
        orig_h=image_h,
        orig_w=image_w,
        crop_size=CROP_SIZE,
        min_circularity=MIN_CIRCULARITY,
        min_area=MIN_AREA,
        filter_edge_cells=FILTER_EDGE_CELLS,
    )

    multi_object_flag = 1 if len(gt_polygons) > 1 else 0

    for item in sample_items:
        img_path = item["img_path"]
        big_label = item["big_label"]
        small_label = item["small_label"]
        obj_idx = item["obj_idx"]

        if obj_idx < 0 or obj_idx >= len(valid_polys):
            rows.append({
                "source_type": "gt",
                "image_path": img_path,
                "json_path": str(json_path),
                "filename": Path(img_path).name,
                "big_label": big_label,
                "small_label": small_label,
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": multi_object_flag,
                "reason": "saved_index_out_of_range",
                "orig_stem": orig_stem,
                "obj_idx": obj_idx,
            })
            continue

        poly_item = valid_polys[obj_idx]

        crop_img = cv2.imread(img_path)
        if crop_img is None:
            rows.append({
                "source_type": "gt",
                "image_path": img_path,
                "json_path": str(json_path),
                "filename": Path(img_path).name,
                "big_label": big_label,
                "small_label": small_label,
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": multi_object_flag,
                "reason": "crop_read_failed",
                "orig_stem": orig_stem,
                "obj_idx": obj_idx,
            })
            continue

        crop_h, crop_w = crop_img.shape[:2]

        contour_local = scale_contour_to_crop(
            contour_base=poly_item["contour_local_base"],
            crop_w=crop_w,
            crop_h=crop_h,
            crop_size=CROP_SIZE,
        )

        contour_local[:, :, 0] = np.clip(contour_local[:, :, 0], 0, crop_w - 1)
        contour_local[:, :, 1] = np.clip(contour_local[:, :, 1], 0, crop_h - 1)

        area = polygon_area(contour_local)
        perimeter = polygon_perimeter(contour_local)
        circularity = polygon_circularity(area, perimeter)

        valid = 1
        reason = "ok"
        if area < 10 or perimeter <= 1e-6:
            valid = 0
            reason = "invalid_geometry"

        rows.append({
            "source_type": "gt",
            "image_path": img_path,
            "json_path": str(json_path),
            "filename": Path(img_path).name,
            "big_label": big_label,
            "small_label": small_label,
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "valid": int(valid),
            "multi_object_flag": int(multi_object_flag),
            "reason": reason,
            "orig_stem": orig_stem,
            "obj_idx": obj_idx,
        })

    return rows


# =========================================================
# QA 复算
# =========================================================

def build_one_row_for_qa(row, split_cfg):
    img_path = row["image_path"]
    json_path = row["json_path"]
    obj_idx = int(row["obj_idx"])

    if not os.path.exists(json_path):
        return None

    try:
        gt_polygons, image_w, image_h = load_ground_truth_polygons(json_path)
    except Exception:
        return None

    if image_w is None or image_h is None:
        return None

    valid_polys = build_valid_gt_detections(
        gt_polygons=gt_polygons,
        orig_h=image_h,
        orig_w=image_w,
        crop_size=CROP_SIZE,
        min_circularity=MIN_CIRCULARITY,
        min_area=MIN_AREA,
        filter_edge_cells=FILTER_EDGE_CELLS,
    )

    if obj_idx < 0 or obj_idx >= len(valid_polys):
        return None

    poly_item = valid_polys[obj_idx]

    crop_img = cv2.imread(img_path)
    if crop_img is None:
        return None

    crop_h, crop_w = crop_img.shape[:2]

    contour_local = scale_contour_to_crop(
        contour_base=poly_item["contour_local_base"],
        crop_w=crop_w,
        crop_h=crop_h,
        crop_size=CROP_SIZE,
    )

    contour_local[:, :, 0] = np.clip(contour_local[:, :, 0], 0, crop_w - 1)
    contour_local[:, :, 1] = np.clip(contour_local[:, :, 1], 0, crop_h - 1)

    return {
        "img_bgr": crop_img,
        "contour_local": contour_local,
    }


# =========================================================
# 构建任务
# =========================================================

def build_group_tasks(split_cfg):
    samples = read_label_txt(split_cfg["label_txt"])
    grouped = defaultdict(list)

    for img_path, big_label, small_label in samples:
        try:
            orig_stem, obj_idx, suffix = parse_singlecell_name(img_path)
        except Exception:
            grouped[("__PARSE_FAILED__", img_path)].append({
                "img_path": img_path,
                "big_label": big_label,
                "small_label": small_label,
                "obj_idx": -1,
            })
            continue

        patient = Path(img_path).parent.name
        group_key = (patient, orig_stem)
        grouped[group_key].append({
            "img_path": img_path,
            "big_label": big_label,
            "small_label": small_label,
            "obj_idx": obj_idx,
        })

    tasks = []

    for key, sample_items in grouped.items():
        if key[0] == "__PARSE_FAILED__":
            tasks.append({
                "patient": "",
                "orig_stem": "",
                "json_path": None,
                "orig_image_path": None,
                "sample_items": sample_items,
            })
            continue

        patient, orig_stem = key
        json_path = str(Path(split_cfg["orig_root"]) / patient / f"{orig_stem}.json")
        if not os.path.exists(json_path):
            json_path = None

        orig_image_path = resolve_image_path(split_cfg["orig_root"], patient, orig_stem)

        tasks.append({
            "patient": patient,
            "orig_stem": orig_stem,
            "json_path": json_path,
            "orig_image_path": orig_image_path,
            "sample_items": sample_items,
        })

    return tasks


# =========================================================
# 主处理
# =========================================================

def process_split(split_cfg):
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(split_cfg["qa_dir"], exist_ok=True)

    tasks = build_group_tasks(split_cfg)
    print(f"\n[Split={split_cfg['name']}] 共 {len(tasks)} 个原图组待处理")
    print(f"[Split={split_cfg['name']}] 使用 {MAX_WORKERS} 个进程并行")

    all_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_group_task, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_cfg['name']}", ncols=100):
            rows = fut.result()
            all_rows.extend(rows)

    all_rows = sorted(all_rows, key=lambda x: x["image_path"])

    df = pd.DataFrame(all_rows)
    keep_cols = [
        "source_type",
        "image_path",
        "json_path",
        "filename",
        "big_label",
        "small_label",
        "area",
        "perimeter",
        "circularity",
        "valid",
        "multi_object_flag",
        "reason",
        "orig_stem",
        "obj_idx",
    ]
    df = df[keep_cols]
    df.to_csv(split_cfg["out_csv"], index=False, encoding="utf-8")

    print(f"[Done] csv saved to: {split_cfg['out_csv']}")
    print(df.head())

    # QA
    qa_candidates = df[df["reason"] == "ok"].head(split_cfg["qa_max"]).to_dict("records")
    print(f"[Split={split_cfg['name']}] 开始生成 QA 图，共 {len(qa_candidates)} 张")

    for i, row in enumerate(tqdm(qa_candidates, desc=f"QA {split_cfg['name']}", ncols=100)):
        qa = build_one_row_for_qa(row, split_cfg)
        if qa is None:
            continue

        qa_name = f"{i:04d}_{Path(row['filename']).stem}.png"
        qa_path = Path(split_cfg["qa_dir"]) / qa_name

        save_qa_image(
            qa["img_bgr"],
            qa["contour_local"],
            qa_path,
            [
                f"valid={int(row['valid'])} multi={int(row['multi_object_flag'])}",
                f"area={float(row['area']):.1f}",
                f"peri={float(row['perimeter']):.1f}",
                f"circ={float(row['circularity']):.4f}",
            ],
        )

    print(f"[Done] qa images saved to: {split_cfg['qa_dir']}")


def main():
    for split_cfg in SPLITS:
        process_split(split_cfg)


if __name__ == "__main__":
    main()