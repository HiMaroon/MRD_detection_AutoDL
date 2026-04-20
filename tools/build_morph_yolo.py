import os
import re
import math
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

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
    # {
    #     "name": "train",
    #     "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train_labels.txt",
    #     "seg_root": "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/train",
    #     "orig_root": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Train",
    #     "out_csv": f"{OUT_ROOT}/train_morph.csv",
    #     "qa_dir": f"{OUT_ROOT}/qa_train_yolo_3feat",
    #     "qa_max": QA_MAX_TRAIN,
    # },
    # {
    #     "name": "val",
    #     "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val_labels.txt",
    #     "seg_root": "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/val",
    #     "orig_root": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Val",
    #     "out_csv": f"{OUT_ROOT}/val_morph.csv",
    #     "qa_dir": f"{OUT_ROOT}/qa_val_yolo_3feat",
    #     "qa_max": QA_MAX_VAL,
    # },
    {
        "name": "test_BJH",
        "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH_labels_16.txt",
        "seg_root": "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_BJH",
        "orig_root": "/root/autodl-tmp/data/BJH_imgs_260211",
        "out_csv": f"{OUT_ROOT}/test_BJH_morph.csv",
        "qa_dir": f"{OUT_ROOT}/qa_test_BJH_yolo_3feat",
        "qa_max": QA_MAX_VAL,
    },
    {
        "name": "test_FXH_noALL",
        "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL_labels_16.txt",
        "seg_root": "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_FXH_noALL",
        "orig_root": "/root/autodl-tmp/data/FXH_imgs_noALL_260318",
        "out_csv": f"{OUT_ROOT}/test_FXH_noALL_morph.csv",
        "qa_dir": f"{OUT_ROOT}/qa_test_FXH_noALL_yolo_3feat",
        "qa_max": QA_MAX_VAL,
    },
    {
        "name": "test_TJMU",
        "label_txt": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU_labels_16.txt",
        "seg_root": "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_TJMU",
        "orig_root": "/root/autodl-tmp/data/TJMU_imgs_260416",
        "out_csv": f"{OUT_ROOT}/test_TJMU_morph.csv",
        "qa_dir": f"{OUT_ROOT}/qa_test_TJMU_yolo_3feat",
        "qa_max": QA_MAX_VAL,
    },
]


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
    stem = Path(img_path).stem
    m = re.match(r"^(.*)_(\d+)_([A-Za-z0-9]+)$", stem)
    if m is None:
        raise ValueError(f"无法解析单细胞文件名: {stem}")
    orig_stem = m.group(1)
    obj_idx = int(m.group(2))
    suffix = m.group(3)
    return orig_stem, obj_idx, suffix


def infer_patient_from_orig_stem(orig_stem: str):
    parts = orig_stem.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return parts[0]


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


def resolve_seg_json_path(seg_root, patient, orig_stem):
    patient_dir = Path(seg_root) / patient
    p = patient_dir / f"{orig_stem}.json"
    if p.exists():
        return str(p)

    found = list(Path(seg_root).rglob(f"{orig_stem}.json"))
    if len(found) > 0:
        return str(found[0])

    return None


def load_seg_json(seg_json_path):
    with open(seg_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_contour_from_segments(cell_data):
    if (
        "segments" in cell_data
        and "x" in cell_data["segments"]
        and "y" in cell_data["segments"]
    ):
        x_coords = cell_data["segments"]["x"]
        y_coords = cell_data["segments"]["y"]
        points = np.array(
            [[int(x), int(y)] for x, y in zip(x_coords, y_coords)],
            dtype=np.int32
        )
        return points.reshape(-1, 1, 2)
    return None


def is_cell_complete(contour, crop_region):
    if contour is None:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    margin = 10

    if (
        x <= crop_x1 + margin
        or x + w >= crop_x2 - margin
        or y <= crop_y1 + margin
        or y + h >= crop_y2 - margin
    ):
        return False

    return True


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


def build_valid_detections(seg_data, orig_w, orig_h,
                           crop_size=576,
                           min_circularity=0.65,
                           min_area=10000,
                           filter_edge_cells=True):
    valid_dets = []

    for i, cell_data in enumerate(seg_data):
        box = cell_data.get("box", {})
        x1 = box.get("x1", 0)
        y1 = box.get("y1", 0)
        x2 = box.get("x2", 0)
        y2 = box.get("y2", 0)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

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

        raw_contour = get_contour_from_segments(cell_data)
        if raw_contour is None:
            continue

        contour_local = raw_contour.copy().astype(np.float32)
        contour_local[:, :, 0] -= crop_x1_img
        contour_local[:, :, 1] -= crop_y1_img

        crop_region = (0, 0, crop_size, crop_size)

        if filter_edge_cells:
            if not is_cell_complete(contour_local, crop_region):
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

        valid_dets.append({
            "raw_det_idx": i,
            "saved_idx": len(valid_dets),
            "contour_local_base": contour_local.astype(np.float32),
        })

    return valid_dets


def process_group_task(task):
    patient = task["patient"]
    orig_stem = task["orig_stem"]
    seg_json_path = task["seg_json_path"]
    orig_image_path = task["orig_image_path"]
    sample_items = task["sample_items"]

    rows = []

    if seg_json_path is None or (not os.path.exists(seg_json_path)):
        for item in sample_items:
            rows.append({
                "source_type": "yolo",
                "image_path": item["img_path"],
                "json_path": str(seg_json_path) if seg_json_path else "",
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": "seg_json_not_found",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    if orig_image_path is None or (not os.path.exists(orig_image_path)):
        for item in sample_items:
            rows.append({
                "source_type": "yolo",
                "image_path": item["img_path"],
                "json_path": str(seg_json_path),
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
        seg_data = load_seg_json(seg_json_path)
    except Exception as e:
        for item in sample_items:
            rows.append({
                "source_type": "yolo",
                "image_path": item["img_path"],
                "json_path": str(seg_json_path),
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": f"seg_json_load_failed:{e}",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    orig_img = cv2.imread(orig_image_path)
    if orig_img is None:
        for item in sample_items:
            rows.append({
                "source_type": "yolo",
                "image_path": item["img_path"],
                "json_path": str(seg_json_path),
                "filename": Path(item["img_path"]).name,
                "big_label": item["big_label"],
                "small_label": item["small_label"],
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "valid": 0,
                "multi_object_flag": 0,
                "reason": "orig_image_read_failed",
                "orig_stem": orig_stem,
                "obj_idx": item["obj_idx"],
            })
        return rows

    orig_h, orig_w = orig_img.shape[:2]

    valid_dets = build_valid_detections(
        seg_data=seg_data,
        orig_w=orig_w,
        orig_h=orig_h,
        crop_size=CROP_SIZE,
        min_circularity=MIN_CIRCULARITY,
        min_area=MIN_AREA,
        filter_edge_cells=FILTER_EDGE_CELLS,
    )

    multi_object_flag = 1 if len(seg_data) > 1 else 0

    for item in sample_items:
        img_path = item["img_path"]
        big_label = item["big_label"]
        small_label = item["small_label"]
        obj_idx = item["obj_idx"]

        if obj_idx < 0 or obj_idx >= len(valid_dets):
            rows.append({
                "source_type": "yolo",
                "image_path": img_path,
                "json_path": str(seg_json_path),
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

        det = valid_dets[obj_idx]

        crop_img = cv2.imread(img_path)
        if crop_img is None:
            rows.append({
                "source_type": "yolo",
                "image_path": img_path,
                "json_path": str(seg_json_path),
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
            contour_base=det["contour_local_base"],
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
            "source_type": "yolo",
            "image_path": img_path,
            "json_path": str(seg_json_path),
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


def build_one_row_for_qa(row, split_cfg):
    img_path = row["image_path"]
    seg_json_path = row["json_path"]
    obj_idx = int(row["obj_idx"])

    if not os.path.exists(seg_json_path):
        return None

    try:
        seg_data = load_seg_json(seg_json_path)
    except Exception:
        return None

    try:
        orig_stem, _, _ = parse_singlecell_name(img_path)
    except Exception:
        return None

    patient = infer_patient_from_orig_stem(orig_stem)
    orig_image_path = resolve_image_path(split_cfg["orig_root"], patient, orig_stem)
    if orig_image_path is None:
        return None

    orig_img = cv2.imread(orig_image_path)
    if orig_img is None:
        return None

    orig_h, orig_w = orig_img.shape[:2]

    valid_dets = build_valid_detections(
        seg_data=seg_data,
        orig_w=orig_w,
        orig_h=orig_h,
        crop_size=CROP_SIZE,
        min_circularity=MIN_CIRCULARITY,
        min_area=MIN_AREA,
        filter_edge_cells=FILTER_EDGE_CELLS,
    )

    if obj_idx < 0 or obj_idx >= len(valid_dets):
        return None

    det = valid_dets[obj_idx]

    crop_img = cv2.imread(img_path)
    if crop_img is None:
        return None

    crop_h, crop_w = crop_img.shape[:2]

    contour_local = scale_contour_to_crop(
        contour_base=det["contour_local_base"],
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

        patient = infer_patient_from_orig_stem(orig_stem)
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
                "seg_json_path": None,
                "orig_image_path": None,
                "sample_items": sample_items,
            })
            continue

        patient, orig_stem = key
        seg_json_path = resolve_seg_json_path(split_cfg["seg_root"], patient, orig_stem)
        orig_image_path = resolve_image_path(split_cfg["orig_root"], patient, orig_stem)

        tasks.append({
            "patient": patient,
            "orig_stem": orig_stem,
            "seg_json_path": seg_json_path,
            "orig_image_path": orig_image_path,
            "sample_items": sample_items,
        })

    return tasks


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
    print(df["reason"].value_counts(dropna=False).head(10))

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