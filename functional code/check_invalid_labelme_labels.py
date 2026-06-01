#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check invalid labels in LabelMe JSON files, generate CSV reports,
highlight bad-labeled cells on images, and copy files needing correction.

功能：
1. 递归扫描一个或多个 INPUT_ROOTS 下所有 LabelMe json。
2. 统计 label 不在 cell_dict 中的 polygon 标注。
3. 输出完整异常明细表、按 label 汇总表、按 json 文件汇总表、扫描总表。
4. 按原始相对路径生成高亮示意图。
   - 同一张图 / 同一个 json 中若有多个异常 label，会全部画在同一张 QC 图中。
   - 每个异常细胞旁边会直接标注：Wrong label: <错误类别>。
5. 按原始相对路径复制需要修改的原图和原 json 文件。

使用方式：
直接修改下面 CONFIG 区域中的 INPUT_ROOTS 和 OUTPUT_DIR，然后运行：
python check_invalid_labelme_labels.py

建议：
OUTPUT_DIR 不要放在原始数据文件夹内部，避免第二次运行时把复制出来的 json 再扫进去。
"""

# ============================================================
# CONFIG：只需要改这里，不需要命令行输入
# ============================================================
INPUT_ROOTS = [
    "/root/autodl-tmp/data/BJH_imgs_260211",
    "/root/autodl-tmp/data/MAIN_imgs_260323_modified",
    "/root/autodl-tmp/data/FXH_imgs_260318",
    "/root/autodl-tmp/data/TJMU_imgs_260416",

]

OUTPUT_DIR = "/root/autodl-tmp/data/bad_label_qc"
MAX_VIS_SIDE = 2048
STRICT_RAW_MATCH = True
COPY_FILES = True
DRAW_VISUALS = True
IMAGE_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
]
# ============================================================



import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "This script requires opencv-python. Install it with: pip install opencv-python"
    ) from e


# =========================
# 1. 合法标签定义
# =========================
# 注意：你原文中 "0":36 后面是中文逗号，这里已修正为英文逗号。
# 另外按你的要求，"1" 也被视为合法标签。
cell_dict = {
    "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
    "E": 8, "B": 9, "M0": 10, "M": 11, "M1": 12, "M2": 13,
    "R": 14, "R1": 15, "R2": 16, "R3": 17,
    "J": 18, "J1": 19, "J2": 20, "J3": 21, "J4": 22,
    "L": 23, "L1": 24, "L2": 25, "L3": 26, "L4": 27,
    "P": 28, "P1": 29, "P2": 30, "P3": 31,
    "B1": 32, "E1": 33, "A": 34, "F": 35, "V": 36, "0": 36, "1": 36,
}
VALID_LABELS = set(cell_dict.keys())

IMAGE_EXTS_DEFAULT = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
]


# =========================
# 3. 文件与图像 IO
# =========================
def safe_relpath(path: Path, root: Path) -> str:
    """Return POSIX-style relative path; fallback to file name if path is outside root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Read image path that may contain Chinese/non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray, ext: str = ".jpg", jpg_quality: int = 95) -> bool:
    """Write image path that may contain Chinese/non-ASCII characters."""
    ensure_parent(path)
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        ext = ".jpg"
    params: List[int] = []
    if ext in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def convert_to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGBA/16-bit image to BGR uint8 for visualization."""
    if img is None:
        raise ValueError("image is None")

    # 16-bit or float -> uint8 by min-max scaling
    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin) * 255.0
        else:
            arr = np.zeros_like(arr)
        img = np.clip(arr, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def find_image_for_json(json_path: Path, data: Dict[str, Any], image_exts: Iterable[str]) -> Optional[Path]:
    """Find the corresponding image for a LabelMe JSON file."""
    json_dir = json_path.parent
    candidates: List[Path] = []

    image_path_in_json = data.get("imagePath")
    if image_path_in_json:
        p = Path(str(image_path_in_json))
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(json_dir / p)

    # Same stem fallback
    for ext in image_exts:
        candidates.append(json_dir / f"{json_path.stem}{ext}")

    # Deduplicate while preserving order
    seen = set()
    for c in candidates:
        c_norm = str(c)
        if c_norm in seen:
            continue
        seen.add(c_norm)
        if c.exists() and c.is_file():
            return c
    return None


# =========================
# 4. LabelMe shape 解析
# =========================
def is_label_invalid(label_raw: Any, strict_raw_match: bool = True) -> Tuple[bool, str, str, str]:
    """
    Return: invalid?, raw_str, stripped_str, reason

    strict_raw_match=True:
        label 原始字符串必须与合法表完全一致。比如 " N1" 会被判为异常。
    strict_raw_match=False:
        会先 strip 后再判断。比如 " N1" 会被当作 N1。
    """
    raw_str = "" if label_raw is None else str(label_raw)
    stripped = raw_str.strip()

    if strict_raw_match:
        if raw_str in VALID_LABELS:
            return False, raw_str, stripped, "OK"
        if stripped in VALID_LABELS and raw_str != stripped:
            return True, raw_str, stripped, "Label has leading/trailing spaces; stripped label is valid"
        return True, raw_str, stripped, "Label not in cell_dict"

    # Non-strict mode
    if stripped in VALID_LABELS:
        return False, raw_str, stripped, "OK after strip"
    return True, raw_str, stripped, "Label not in cell_dict after strip"


def points_to_array(points: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            return None
        return arr
    except Exception:
        return None


def bbox_from_points(points_arr: Optional[np.ndarray]) -> Tuple[str, str, str, str, str, str]:
    if points_arr is None or points_arr.size == 0:
        return "", "", "", "", "", ""
    xmin = float(np.min(points_arr[:, 0]))
    ymin = float(np.min(points_arr[:, 1]))
    xmax = float(np.max(points_arr[:, 0]))
    ymax = float(np.max(points_arr[:, 1]))
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return (
        f"{xmin:.2f}", f"{ymin:.2f}", f"{xmax:.2f}", f"{ymax:.2f}",
        f"{cx:.2f}", f"{cy:.2f}",
    )


# =========================
# 5. 高亮图绘制
# =========================
def resize_for_visual(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize image for visualization and return scale factor applied to coordinates."""
    if max_side is None or max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_invalid_shapes(
    image_path: Path,
    invalid_shapes: List[Dict[str, Any]],
    output_path: Path,
    max_vis_side: int = 2200,
    alpha: float = 0.35,
) -> bool:
    img_raw = imread_unicode(image_path)
    if img_raw is None:
        return False

    img = convert_to_bgr_uint8(img_raw)
    img, scale = resize_for_visual(img, max_vis_side)

    overlay = img.copy()
    line_thickness = max(2, int(round(max(img.shape[:2]) / 900)))
    font_scale = max(0.55, max(img.shape[:2]) / 1800)
    text_thickness = max(1, int(round(line_thickness / 2)))

    # OpenCV uses BGR. 红色填充 + 黄色边框，便于在细胞图上看清。
    fill_color = (0, 0, 255)
    outline_color = (0, 255, 255)
    bbox_color = (255, 255, 255)
    text_color = (0, 0, 255)
    text_bg_color = (255, 255, 255)

    for item in invalid_shapes:
        points_arr = points_to_array(item.get("points"))
        if points_arr is None:
            continue
        pts = np.round(points_arr * scale).astype(np.int32)

        if pts.shape[0] >= 3:
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(img, [pts], isClosed=True, color=outline_color, thickness=line_thickness)
        else:
            # line/point fallback
            cv2.polylines(img, [pts], isClosed=False, color=outline_color, thickness=line_thickness)

        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, line_thickness)

        shape_index = item.get("shape_index", "")
        label = str(item.get("label_raw", ""))
        text = f"#{shape_index}  Wrong label: {label}"
        tx = max(0, x)
        ty = max(24, y - 6)

        # 避免文本超出右侧边界：如果右侧放不下，自动左移。
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        if tx + tw + 8 > img.shape[1]:
            tx = max(0, img.shape[1] - tw - 8)

        cv2.rectangle(img, (tx, ty - th - baseline - 5), (tx + tw + 8, ty + baseline + 2), text_bg_color, -1)
        cv2.putText(img, text, (tx + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)

    img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # 标题条
    title = f"Invalid labels in this image: {len(invalid_shapes)} | {image_path.name}"
    bar_h = int(max(36, 42 * font_scale))
    canvas = np.full((img.shape[0] + bar_h, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[bar_h:, :, :] = img
    cv2.putText(canvas, title, (10, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

    return imwrite_unicode(output_path, canvas, ext=output_path.suffix or ".jpg")


# =========================
# 6. CSV 输出
# =========================
INVALID_DETAIL_FIELDS = [
    "input_root",
    "relative_dir",
    "json_relpath",
    "json_path",
    "image_relpath",
    "image_path",
    "imagePath_in_json",
    "image_found",
    "imageWidth_in_json",
    "imageHeight_in_json",
    "shape_index",
    "shape_type",
    "label_raw",
    "label_stripped",
    "reason",
    "n_points",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_cx",
    "bbox_cy",
]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_counter_csv(path: Path, counter: Counter, field_label: str = "label") -> None:
    rows = []
    total = sum(counter.values())
    for key, count in counter.most_common():
        pct = count / total * 100 if total > 0 else 0.0
        rows.append({field_label: key, "count": count, "percentage": f"{pct:.4f}%"})
    write_csv(path, rows, [field_label, "count", "percentage"])


def write_json_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "input_root", "relative_dir", "json_relpath", "json_path", "image_relpath", "image_path",
        "invalid_count", "invalid_labels", "visual_path", "copied_json_path", "copied_image_path",
    ]
    write_csv(path, rows, fields)


def write_scan_summary(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    fields = [
        "input_root", "total_json_files", "json_files_with_invalid_labels", "total_shapes_checked",
        "total_invalid_shapes", "total_json_read_errors", "total_missing_images_for_invalid_json",
        "visuals_generated", "files_copied_pairs_or_jsons",
    ]
    write_csv(path, summary_rows, fields)


# =========================
# 7. 主流程
# =========================
def scan_one_root(
    input_root: Path,
    output_dir: Path,
    image_exts: Iterable[str],
    strict_raw_match: bool,
    max_vis_side: int,
    copy_files: bool,
    draw_visuals: bool,
) -> Dict[str, Any]:
    input_root = input_root.resolve()
    root_name = input_root.name or "root"

    # 多个 input_root 时，为避免同名冲突，输出中保留 root_name
    report_dir = output_dir / root_name / "reports"
    visual_root = output_dir / root_name / "invalid_label_QC_visuals"
    copy_root = output_dir / root_name / "needs_fix_original_files"

    invalid_detail_rows: List[Dict[str, Any]] = []
    json_summary_rows: List[Dict[str, Any]] = []
    json_error_rows: List[Dict[str, Any]] = []

    invalid_label_counter: Counter = Counter()
    invalid_json_counter: Counter = Counter()
    total_json = 0
    total_shapes_checked = 0
    visuals_generated = 0
    copied_items = 0
    missing_images_for_invalid_json = 0

    json_paths = sorted(input_root.rglob("*.json"))

    for json_path in json_paths:
        total_json += 1
        rel_json = safe_relpath(json_path, input_root)
        rel_dir = Path(rel_json).parent.as_posix()
        if rel_dir == ".":
            rel_dir = ""

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            json_error_rows.append({
                "input_root": str(input_root),
                "json_relpath": rel_json,
                "json_path": str(json_path),
                "error": repr(e),
            })
            continue

        image_path = find_image_for_json(json_path, data, image_exts)
        image_relpath = safe_relpath(image_path, input_root) if image_path else ""
        image_found = image_path is not None
        image_path_str = str(image_path) if image_path else ""

        invalid_shapes_in_json: List[Dict[str, Any]] = []
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            json_error_rows.append({
                "input_root": str(input_root),
                "json_relpath": rel_json,
                "json_path": str(json_path),
                "error": "data['shapes'] is not a list",
            })
            continue

        for idx, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue

            # 默认只检查 polygon，因为你的细胞标注是 polygon。
            # 若你想所有 shape 都查，可把这句注释掉。
            shape_type = shape.get("shape_type", "")
            if shape_type != "polygon":
                continue

            total_shapes_checked += 1
            invalid, raw_str, stripped, reason = is_label_invalid(
                shape.get("label"), strict_raw_match=strict_raw_match
            )
            if not invalid:
                continue

            points_arr = points_to_array(shape.get("points"))
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_cx, bbox_cy = bbox_from_points(points_arr)
            n_points = int(points_arr.shape[0]) if points_arr is not None else 0

            detail = {
                "input_root": str(input_root),
                "relative_dir": rel_dir,
                "json_relpath": rel_json,
                "json_path": str(json_path),
                "image_relpath": image_relpath,
                "image_path": image_path_str,
                "imagePath_in_json": data.get("imagePath", ""),
                "image_found": str(image_found),
                "imageWidth_in_json": data.get("imageWidth", ""),
                "imageHeight_in_json": data.get("imageHeight", ""),
                "shape_index": idx,
                "shape_type": shape_type,
                "label_raw": raw_str,
                "label_stripped": stripped,
                "reason": reason,
                "n_points": n_points,
                "bbox_xmin": bbox_xmin,
                "bbox_ymin": bbox_ymin,
                "bbox_xmax": bbox_xmax,
                "bbox_ymax": bbox_ymax,
                "bbox_cx": bbox_cx,
                "bbox_cy": bbox_cy,
            }
            invalid_detail_rows.append(detail)
            invalid_shapes_in_json.append({
                "shape_index": idx,
                "label_raw": raw_str,
                "points": shape.get("points", []),
            })
            invalid_label_counter[raw_str] += 1
            invalid_json_counter[rel_json] += 1

        if not invalid_shapes_in_json:
            continue

        # 生成高亮示意图
        visual_path = ""
        if draw_visuals:
            if image_path is not None:
                # visual 保存为 jpg，按 json 的相对目录结构保存。
                visual_rel_dir = Path(rel_dir) if rel_dir else Path("")
                visual_name = f"{json_path.stem}__BAD_LABELS.jpg"
                visual_out = visual_root / visual_rel_dir / visual_name
                ok = draw_invalid_shapes(
                    image_path=image_path,
                    invalid_shapes=invalid_shapes_in_json,
                    output_path=visual_out,
                    max_vis_side=max_vis_side,
                )
                if ok:
                    visuals_generated += 1
                    visual_path = str(visual_out)
            else:
                missing_images_for_invalid_json += 1

        # 复制原 json 和原图到待修改文件夹
        copied_json_path = ""
        copied_image_path = ""
        if copy_files:
            copy_rel_dir = Path(rel_dir) if rel_dir else Path("")
            dst_json = copy_root / copy_rel_dir / json_path.name
            ensure_parent(dst_json)
            shutil.copy2(json_path, dst_json)
            copied_json_path = str(dst_json)
            copied_items += 1

            if image_path is not None and image_path.exists():
                # 优先按原图文件名复制到同一个相对目录。
                dst_img = copy_root / copy_rel_dir / image_path.name
                ensure_parent(dst_img)
                # 避免 json 和 image 同名不同路径的覆盖极端情况，一般不会发生。
                shutil.copy2(image_path, dst_img)
                copied_image_path = str(dst_img)

        json_summary_rows.append({
            "input_root": str(input_root),
            "relative_dir": rel_dir,
            "json_relpath": rel_json,
            "json_path": str(json_path),
            "image_relpath": image_relpath,
            "image_path": image_path_str,
            "invalid_count": len(invalid_shapes_in_json),
            "invalid_labels": ";".join([str(x[0]) + ":" + str(x[1]) for x in Counter([s["label_raw"] for s in invalid_shapes_in_json]).items()]),
            "visual_path": visual_path,
            "copied_json_path": copied_json_path,
            "copied_image_path": copied_image_path,
        })

    # 输出 CSV
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "invalid_label_instances.csv", invalid_detail_rows, INVALID_DETAIL_FIELDS)
    write_counter_csv(report_dir / "invalid_label_summary_by_label.csv", invalid_label_counter, field_label="invalid_label")
    write_json_summary_csv(report_dir / "invalid_label_summary_by_json.csv", json_summary_rows)

    if json_error_rows:
        write_csv(report_dir / "json_read_errors.csv", json_error_rows, ["input_root", "json_relpath", "json_path", "error"])
    else:
        # 也生成空表，方便确认没有读取错误
        write_csv(report_dir / "json_read_errors.csv", [], ["input_root", "json_relpath", "json_path", "error"])

    summary = {
        "input_root": str(input_root),
        "total_json_files": total_json,
        "json_files_with_invalid_labels": len(json_summary_rows),
        "total_shapes_checked": total_shapes_checked,
        "total_invalid_shapes": len(invalid_detail_rows),
        "total_json_read_errors": len(json_error_rows),
        "total_missing_images_for_invalid_json": missing_images_for_invalid_json,
        "visuals_generated": visuals_generated,
        "files_copied_pairs_or_jsons": copied_items,
    }
    return summary


def main() -> None:
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for root_str in INPUT_ROOTS:
        input_root = Path(root_str).resolve()
        if not input_root.exists() or not input_root.is_dir():
            print(f"[WARN] Skip invalid input root: {input_root}")
            continue

        print(f"\n========== Scanning: {input_root} ==========")
        summary = scan_one_root(
            input_root=input_root,
            output_dir=output_dir,
            image_exts=IMAGE_EXTS,
            strict_raw_match=STRICT_RAW_MATCH,
            max_vis_side=MAX_VIS_SIDE,
            copy_files=COPY_FILES,
            draw_visuals=DRAW_VISUALS,
        )
        summary_rows.append(summary)
        print(
            f"[DONE] {input_root}\n"
            f"  total_json_files = {summary['total_json_files']}\n"
            f"  json_files_with_invalid_labels = {summary['json_files_with_invalid_labels']}\n"
            f"  total_invalid_shapes = {summary['total_invalid_shapes']}\n"
            f"  visuals_generated = {summary['visuals_generated']}\n"
        )

    write_scan_summary(output_dir / "ALL_ROOTS_SCAN_SUMMARY.csv", summary_rows)
    print(f"\nAll done. Output folder: {output_dir}")
    print(f"Global summary: {output_dir / 'ALL_ROOTS_SCAN_SUMMARY.csv'}")


if __name__ == "__main__":
    main()
