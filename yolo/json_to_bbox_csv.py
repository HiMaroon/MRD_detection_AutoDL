#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert YOLO prediction JSON files to bbox CSV used by src/datasets.py.

Output columns:
    image_path,x1,y1,x2,y2,mpp

Assumptions:
1) Prediction JSON files are saved by yolo/yolotest_1.0.py under:
       <pred_root>/<split>/**/*.json
2) The folder structure under split mirrors original image folder structure.
3) Each JSON file contains a list of detections. Each detection is expected to
   include either:
       det["box"]["x1|y1|x2|y2"]  (preferred)
   or
       det["bbox"] / det["xyxy"] with 4 values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


Box = Tuple[float, float, float, float]


DEFAULT_SPLIT_IMAGE_ROOTS = {
    "train": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Train",
    "val": "/root/autodl-tmp/data/MAIN_imgs_split_260323/Val",
    "test_FXH_noALL": "/root/autodl-tmp/data/FXH_imgs_noALL_260318",
    "test_BJH": "/root/autodl-tmp/data/BJH_imgs_260211",
    "test_TJMU": "/root/autodl-tmp/data/TJMU_imgs_260318",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO JSON predictions to bbox CSV.")
    parser.add_argument(
        "--pred-root",
        type=str,
        default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323",
        help="YOLO prediction root directory",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output bbox CSV path",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test_FXH_noALL", "test_BJH", "test_TJMU"],
        help="Splits to process",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="conf",
        choices=["conf", "center"],
        help="How to select one bbox when multiple detections exist",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Optional default mpp value written to CSV",
    )
    parser.add_argument(
        "--allow-empty-full-image",
        action="store_true",
        help="If set, empty detections will fallback to full-image bbox instead of skipping",
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        default=".jpg",
        help="Image extension used when reconstructing image path from prediction JSON",
    )
    parser.add_argument(
        "--split-image-root",
        action="append",
        default=[],
        help='Override split image root: format "split=/abs/path". Can be passed multiple times.',
    )
    return parser.parse_args()


def build_split_roots(overrides: Iterable[str]) -> Dict[str, Path]:
    roots = {k: Path(v) for k, v in DEFAULT_SPLIT_IMAGE_ROOTS.items()}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f'Invalid --split-image-root format: "{item}". Use split=/abs/path')
        split, path = item.split("=", 1)
        split = split.strip()
        path = path.strip()
        if not split or not path:
            raise ValueError(f'Invalid --split-image-root value: "{item}"')
        roots[split] = Path(path)
    return roots


def get_confidence(det: dict) -> float:
    for key in ("confidence", "conf", "score"):
        if key in det and det[key] is not None:
            try:
                return float(det[key])
            except Exception:
                pass
    return -1.0


def parse_box(det: dict) -> Optional[Box]:
    if "box" in det and isinstance(det["box"], dict):
        b = det["box"]
        keys = ("x1", "y1", "x2", "y2")
        if all(k in b for k in keys):
            try:
                return float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            except Exception:
                return None

    for key in ("bbox", "xyxy"):
        if key in det and isinstance(det[key], (list, tuple)) and len(det[key]) >= 4:
            try:
                x1, y1, x2, y2 = det[key][:4]
                return float(x1), float(y1), float(x2), float(y2)
            except Exception:
                return None
    return None


def center_distance_sq(box: Box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (cx - 0.5 * w) ** 2 + (cy - 0.5 * h) ** 2


def select_box(dets: List[dict], strategy: str, image_path: Path) -> Optional[Box]:
    candidates: List[Tuple[Box, dict]] = []
    for det in dets:
        box = parse_box(det)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((box, det))

    if not candidates:
        return None

    if strategy == "conf":
        candidates.sort(key=lambda x: get_confidence(x[1]), reverse=True)
        return candidates[0][0]

    if strategy == "center":
        try:
            with Image.open(image_path) as im:
                w, h = im.size
            candidates.sort(key=lambda x: center_distance_sq(x[0], w, h))
            return candidates[0][0]
        except Exception:
            candidates.sort(key=lambda x: get_confidence(x[1]), reverse=True)
            return candidates[0][0]

    raise ValueError(f"Unsupported strategy: {strategy}")


def reconstruct_image_path(pred_root: Path, split: str, split_root: Path, pred_json_path: Path, img_ext: str) -> Path:
    rel = pred_json_path.relative_to(pred_root / split).with_suffix(img_ext)
    return split_root / rel


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    out_csv = Path(args.out_csv)
    split_roots = build_split_roots(args.split_image_root)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "json_total": 0,
        "json_parse_fail": 0,
        "img_missing": 0,
        "empty_det": 0,
        "rows": 0,
    }

    rows = []
    for split in args.splits:
        pred_split_dir = pred_root / split
        if not pred_split_dir.exists():
            print(f"[WARN] split not found, skip: {pred_split_dir}")
            continue

        if split not in split_roots:
            print(f"[WARN] split image root missing, skip split={split}")
            continue

        json_files = list(pred_split_dir.rglob("*.json"))
        stats["json_total"] += len(json_files)
        print(f"[INFO] split={split}, json_files={len(json_files)}")

        for json_path in json_files:
            try:
                dets = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                stats["json_parse_fail"] += 1
                continue

            if not isinstance(dets, list):
                stats["json_parse_fail"] += 1
                continue

            image_path = reconstruct_image_path(
                pred_root=pred_root,
                split=split,
                split_root=split_roots[split],
                pred_json_path=json_path,
                img_ext=args.img_ext,
            )

            if not image_path.exists():
                stats["img_missing"] += 1
                continue

            box = select_box(dets, strategy=args.strategy, image_path=image_path)
            if box is None:
                stats["empty_det"] += 1
                if not args.allow_empty_full_image:
                    continue
                with Image.open(image_path) as im:
                    w, h = im.size
                box = (0.0, 0.0, float(w), float(h))

            x1, y1, x2, y2 = box
            rows.append(
                {
                    "image_path": str(image_path),
                    "x1": f"{x1:.4f}",
                    "y1": f"{y1:.4f}",
                    "x2": f"{x2:.4f}",
                    "y2": f"{y2:.4f}",
                    "mpp": "" if args.mpp is None else f"{args.mpp:.6f}",
                }
            )
            stats["rows"] += 1

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "x1", "y1", "x2", "y2", "mpp"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Done =====")
    print(f"Output CSV: {out_csv}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

