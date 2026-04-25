import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

"""
YOLO 分割结果 -> 单细胞动态裁剪 + 形态学参数导出。

本版本对齐 build_morph_yolo.py 的形态学字段：
1) 形态学列名与 build_morph_yolo.py 的 ALL_FEATURES 完全一致。
2) 轮廓特征与弱外观特征的计算方式借鉴 build_morph_yolo.py。
3) yolo2singlecell_dynamic_morph.py 原有的非形态学字段保持原样。
"""

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


def imread_chinese(path: str):
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_chinese(image: np.ndarray, path: str) -> bool:
    ok, buffer = cv2.imencode(Path(path).suffix, image)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buffer)
    return True


def get_contour_from_segments(cell_data: Dict) -> Optional[np.ndarray]:
    seg = cell_data.get("segments", {})
    x_coords = seg.get("x")
    y_coords = seg.get("y")
    if not x_coords or not y_coords or len(x_coords) != len(y_coords):
        return None

    points = np.array([[int(x), int(y)] for x, y in zip(x_coords, y_coords)], dtype=np.int32)
    if len(points) < 3:
        return None
    return points.reshape(-1, 1, 2)


def compute_contour_features(contour: np.ndarray) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0

    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    aspect_ratio = float(w / h) if h > 0 else 0.0
    extent = float(area / (w * h)) if (w > 0 and h > 0) else 0.0

    hull = cv2.convexHull(contour.astype(np.float32))
    convex_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(area / convex_area) if convex_area > 1e-6 else 0.0

    equiv_diameter = float(math.sqrt(4.0 * area / math.pi)) if area > 0 else 0.0
    major_axis_length, minor_axis_length, eccentricity = 0.0, 0.0, 0.0

    if len(contour) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(contour.astype(np.float32))
        major_axis_length = float(max(ma, mi))
        minor_axis_length = float(min(ma, mi))
        if major_axis_length > 1e-6:
            eccentricity = float(
                math.sqrt(max(0.0, 1.0 - (minor_axis_length ** 2) / (major_axis_length ** 2)))
            )

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


def masked_stats(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    vals = arr[mask > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())


def entropy_8bit(gray_u8: np.ndarray, mask: np.ndarray) -> float:
    vals = gray_u8[mask > 0]
    if vals.size == 0:
        return 0.0
    hist = cv2.calcHist([vals.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def glcm_like_texture(gray_f32: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    # 与 build_morph_yolo.py 保持一致的轻量纹理近似
    if gray_f32.shape[1] < 2:
        return 0.0, 0.0
    diff = np.abs(gray_f32[:, 1:] - gray_f32[:, :-1])
    m2 = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    v = diff[m2]
    if v.size == 0:
        return 0.0, 0.0
    energy = float(np.mean((1.0 - np.clip(v, 0, 1)) ** 2))
    contrast = float(np.mean(v ** 2))
    return energy, contrast


def compute_weak_appearance_features(crop_bgr: np.ndarray, contour: np.ndarray) -> Dict[str, float]:
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


def resolve_dynamic_crop_size(
    contour: np.ndarray,
    scale_factor: float,
    min_crop_size: int,
    max_crop_size: int,
) -> int:
    _, _, w, h = cv2.boundingRect(contour)
    roi_size = max(int(w), int(h))
    dynamic_size = int(round(roi_size * scale_factor))
    dynamic_size = max(min_crop_size, dynamic_size)
    dynamic_size = min(max_crop_size, dynamic_size)
    return dynamic_size


def compute_square_crop(center_x: float, center_y: float, crop_size: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
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


def is_cell_complete(contour_local: np.ndarray, crop_size: int, margin: int = 10) -> bool:
    x, y, w, h = cv2.boundingRect(contour_local.astype(np.int32))
    if x <= margin or y <= margin:
        return False
    if x + w >= crop_size - margin or y + h >= crop_size - margin:
        return False
    return True


def scale_contour(contour: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    contour = contour.copy().astype(np.float32)
    if src_w > 0:
        contour[:, :, 0] *= dst_w / float(src_w)
    if src_h > 0:
        contour[:, :, 1] *= dst_h / float(src_h)
    contour[:, :, 0] = np.clip(contour[:, :, 0], 0, max(dst_w - 1, 0))
    contour[:, :, 1] = np.clip(contour[:, :, 1], 0, max(dst_h - 1, 0))
    return contour


def process_cells_dynamic(
    segmentation_json_path: Path,
    image_path: Path,
    output_dir: Path,
    remove_background: bool = False,
    filter_edge_cells: bool = True,
    min_circularity: float = 0.65,
    min_area: float = 10000,
    min_crop_size: int = 256,
    max_crop_size: int = 768,
    crop_scale_factor: float = 1.2,
    output_size: Optional[int] = None,
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = imread_chinese(str(image_path))
    if image is None:
        return []

    with open(segmentation_json_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    rows: List[Dict] = []
    saved_idx = 0
    for cell_data in seg_data:
        contour_global = get_contour_from_segments(cell_data)
        if contour_global is None:
            continue

        # 过滤仍然沿用原始图坐标上的几何约束
        precheck_morph = compute_contour_features(contour_global)
        if precheck_morph["area"] < min_area:
            continue
        if precheck_morph["circularity"] < min_circularity:
            continue

        crop_size = resolve_dynamic_crop_size(
            contour=contour_global,
            scale_factor=crop_scale_factor,
            min_crop_size=min_crop_size,
            max_crop_size=max_crop_size,
        )

        box = cell_data.get("box", {})
        x1 = float(box.get("x1", 0))
        y1 = float(box.get("y1", 0))
        x2 = float(box.get("x2", 0))
        y2 = float(box.get("y2", 0))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        crop_x1, crop_y1, crop_x2, crop_y2 = compute_square_crop(
            center_x, center_y, crop_size, image.shape[1], image.shape[0]
        )

        contour_local = contour_global.copy().astype(np.float32)
        contour_local[:, :, 0] -= crop_x1
        contour_local[:, :, 1] -= crop_y1

        if filter_edge_cells and (not is_cell_complete(contour_local, crop_size)):
            continue

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop_img.shape[:2]
        contour_for_saved = contour_local.copy()

        if remove_background:
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour_for_saved.astype(np.int32)], 255)
            crop_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)

        if output_size is not None and output_size > 0:
            crop_img = cv2.resize(crop_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
            contour_for_saved = scale_contour(
                contour_for_saved,
                src_w=crop_w,
                src_h=crop_h,
                dst_w=output_size,
                dst_h=output_size,
            )

        # 对齐 build_morph_yolo.py：在最终保存图像坐标系下计算特征
        contour_feat = compute_contour_features(contour_for_saved)
        appear_feat = compute_weak_appearance_features(crop_img, contour_for_saved)
        morph = {**contour_feat, **appear_feat}

        image_stem = image_path.stem
        out_name = f"{image_stem}_{saved_idx:03d}_0.png"
        out_path = output_dir / out_name
        if not imwrite_chinese(crop_img, str(out_path)):
            continue

        row = {
            "filename": out_name,
            "image_path": str(out_path),
            "source_image": str(image_path),
            "source_seg_json": str(segmentation_json_path),
            "crop_size": int(crop_size),
            "crop_scale_factor": float(crop_scale_factor),
            "crop_x1": int(crop_x1),
            "crop_y1": int(crop_y1),
            "crop_x2": int(crop_x2),
            "crop_y2": int(crop_y2),
        }
        row.update({k: float(morph.get(k, 0.0)) for k in ALL_FEATURES})
        rows.append(row)
        saved_idx += 1

    return rows


def _build_stem_index(root_dir: Path, exts: List[str]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for ext in exts:
        for p in root_dir.rglob(f"*{ext}"):
            idx.setdefault(p.stem, p)
    return idx


def batch_process_dynamic(
    seg_json_dir: Path,
    image_dir: Path,
    output_dir: Path,
    morph_csv_path: Path,
    **kwargs,
):
    seg_json_files = list(seg_json_dir.rglob("*.json"))
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_stem_idx = _build_stem_index(image_dir, image_exts)

    all_rows: List[Dict] = []
    missing_images = 0

    for seg_json_file in tqdm(seg_json_files, desc="动态裁剪中"):
        image_file = image_stem_idx.get(seg_json_file.stem)
        if image_file is None:
            missing_images += 1
            continue

        rows = process_cells_dynamic(
            segmentation_json_path=seg_json_file,
            image_path=image_file,
            output_dir=output_dir,
            **kwargs,
        )
        all_rows.extend(rows)

    if missing_images > 0:
        print(f"⚠️ 缺少匹配图像：{missing_images} 个")

    if len(all_rows) == 0:
        print("没有可写入的形态学记录。")
        return

    morph_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_fields = [
        "filename", "image_path", "source_image", "source_seg_json",
        "crop_size", "crop_scale_factor", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    ]
    fieldnames = fixed_fields + ALL_FEATURES

    with open(morph_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"完成：保存单细胞 {len(all_rows)} 张，形态学 CSV -> {morph_csv_path}")


def main():
    # ===== 参数区（按 yolo/yolo2singlecell.py 风格在此直接修改）=====
    global_cfg = {
        "remove_background": False,
        "filter_edge_cells": True,
        "min_circularity": 0.65,
        "min_area": 10000,
        "min_crop_size": 100,
        "max_crop_size": 600,
        "crop_scale_factor": 1.1,  # 动态裁剪：ROI 的 1.2 倍
        "output_size": 224,        # 若不想缩放可设为 None
    }

    datasets = [
        {
            "name": "train",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/train",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260323/Train",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_dynamic_size1.1_260323/train",
        },
        {
            "name": "val",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/val",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260323/Val",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_dynamic_size1.1_260323/val",
        },
        # {
        #     "name": "test_TJMU",
        #     "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_TJMU",
        #     "img": r"/root/autodl-tmp/data/TJMU_imgs_260416",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_dynamic_260323/test_TJMU",
        # },
    ]

    for ds in datasets:
        print(f"开始处理数据集：{ds['name']}")
        out_dir = Path(ds["out"])

        # 你指定的 CSV 保存目录
        morph_csv_dir = Path("/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_dynamic_size1.1_260323/morph_csv")
        morph_csv_dir.mkdir(parents=True, exist_ok=True)

        # 文件名里带 name 字段
        morph_csv_path = morph_csv_dir / f"morphology_{ds['name']}.csv"

        batch_process_dynamic(
            seg_json_dir=Path(ds["seg"]),
            image_dir=Path(ds["img"]),
            output_dir=out_dir,
            morph_csv_path=morph_csv_path,
            **global_cfg,
        )


if __name__ == "__main__":
    main()
