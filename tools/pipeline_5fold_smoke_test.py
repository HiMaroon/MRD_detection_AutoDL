"""
小规模冒烟测试：验证 tools/pipeline_5fold_autorun.py 的关键前处理环节。

覆盖模块：
1) 患者级 fold 划分
2) 患者目录实体化（非软链接）
3) Labelme -> YOLO 数据集生成
4) GT -> 单细胞提取（fast_gt2singlecell）
5) 标签 txt 生成（DataTxtGenerator）

注意：该脚本不做 YOLO 训练/推理，也不做分类模型训练，仅用于快速验证流水线核心组件可运行。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.DataTextGenerator_test import DataTxtGenerator
from tools.pipeline_5fold_autorun import (
    build_folds,
    fast_gt2singlecell,
    json_to_yolo_dataset,
    materialize_patient_dir,
)


def _make_one_patient_folder(root: Path, patient_id: str):
    pdir = root / patient_id
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) 图片
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :, 1] = 120
    img_path = pdir / f"{patient_id}-10_000_N.jpg"
    Image.fromarray(img).save(img_path)

    # 2) labelme json（同时用于 yolo 转换和 gt 单细胞提取）
    # shape_type=polygon，label='N'，便于后续单细胞输出包含类别后缀
    ann = {
        "imagePath": img_path.name,
        "imageHeight": 128,
        "imageWidth": 128,
        "shapes": [
            {
                "label": "N",
                "shape_type": "polygon",
                "points": [[30, 30], [100, 30], [100, 100], [30, 100]],
            }
        ],
    }
    with open(pdir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)


def run_smoke_test():
    with tempfile.TemporaryDirectory(prefix="pipeline_5fold_smoke_") as td:
        work = Path(td)
        img_root = work / "MAIN_imgs_260323"
        out_root = work / "runs_5fold_smoke"
        out_root.mkdir(parents=True, exist_ok=True)

        # ---- 构造最小患者数据（2HC + 2AML，满足 2-fold 分层）----
        rows = [
            {"正式编号": "PKUPH-001", "患者大类型": "HC"},
            {"正式编号": "BEPH-002", "患者大类型": "HC"},
            {"正式编号": "TAB-003", "患者大类型": "AML"},
            {"正式编号": "PKUPH-004", "患者大类型": "AML"},
        ]
        for r in rows:
            _make_one_patient_folder(img_root, r["正式编号"])

        excel_path = work / "patient_data_smoke.xlsx"
        pd.DataFrame(rows).to_excel(excel_path, sheet_name="总表", index=False)

        # ---- 1) fold 划分 ----
        base_df, folds, _ = build_folds(
            excel_path=excel_path,
            sheet_name="总表",
            img_root=img_root,
            n_splits=2,
            random_state=42,
        )
        assert len(folds) == 2, "期望得到2个fold"
        assert len(base_df) == 4, "期望4个有效患者"

        # 仅跑 fold1 的前处理冒烟
        fold = folds[0]
        fold_root = out_root / "fold1"
        train_dir = fold_root / "raw_split" / "Train"
        val_dir = fold_root / "raw_split" / "Val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # ---- 2) 实体化目录（不用软链接）----
        for pid in fold.train_ids:
            materialize_patient_dir(img_root / pid, train_dir / pid)
        for pid in fold.val_ids:
            materialize_patient_dir(img_root / pid, val_dir / pid)

        # ---- 3) Labelme -> YOLO ----
        yolo_dataset = fold_root / "yolo_dataset"
        json_to_yolo_dataset(train_dir, val_dir, yolo_dataset)
        assert (yolo_dataset / "dataset.yaml").exists(), "dataset.yaml 未生成"

        # ---- 4) GT -> 单细胞 ----
        singlecell_root = fold_root / "singlecell"
        fast_gt2singlecell(
            points_json_dir=train_dir,
            output_base_dir=singlecell_root / "train_groundtruth",
            remove_background=False,
            filter_edge_cells=False,
            min_circularity=0.0,
            min_area=1,
            crop_size=64,
            output_size=64,
            max_workers=2,
        )
        fast_gt2singlecell(
            points_json_dir=val_dir,
            output_base_dir=singlecell_root / "val_groundtruth",
            remove_background=False,
            filter_edge_cells=False,
            min_circularity=0.0,
            min_area=1,
            crop_size=64,
            output_size=64,
            max_workers=2,
        )

        # ---- 5) 标签 txt 生成 ----
        # DataTxtGenerator 期望存在这些 split 目录；这里预先创建，避免“目录不存在”的噪声日志。
        expected_splits = ["train", "val", "test_BJH", "test_TJMU", "test_FXH_noALL"]
        for split in expected_splits:
            (singlecell_root / split).mkdir(parents=True, exist_ok=True)

        # 为了验证 train/val 标签文件也能生成，这里从 GT 结果中拷贝少量样本到 yolo train/val 目录
        train_gt_samples = list((singlecell_root / "train_groundtruth").rglob("*.png"))
        val_gt_samples = list((singlecell_root / "val_groundtruth").rglob("*.png"))
        if train_gt_samples:
            (singlecell_root / "train" / train_gt_samples[0].name).write_bytes(train_gt_samples[0].read_bytes())
        if val_gt_samples:
            (singlecell_root / "val" / val_gt_samples[0].name).write_bytes(val_gt_samples[0].read_bytes())

        DataTxtGenerator(str(singlecell_root))
        train_gt_txt = singlecell_root / "train_groundtruth_labels.txt"
        val_gt_txt = singlecell_root / "val_groundtruth_labels.txt"
        train_txt = singlecell_root / "train_labels.txt"
        val_txt = singlecell_root / "val_labels.txt"
        assert train_gt_txt.exists(), "train_groundtruth_labels.txt 未生成"
        assert val_gt_txt.exists(), "val_groundtruth_labels.txt 未生成"
        assert train_txt.exists(), "train_labels.txt 未生成"
        assert val_txt.exists(), "val_labels.txt 未生成"

        print("=" * 80)
        print("✅ 冒烟测试通过")
        print(f"工作目录: {work}")
        print(f"fold1 train患者数: {len(fold.train_ids)}")
        print(f"fold1 val患者数:   {len(fold.val_ids)}")
        print(f"YOLO dataset: {yolo_dataset}")
        print(f"SingleCell root: {singlecell_root}")
        print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
