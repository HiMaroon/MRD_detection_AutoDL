import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from ultralytics import YOLO

from tools.pipeline_5fold_autorun import (
    FoldInfo,
    build_folds,
    load_folds_from_assignment,
    ensure_clean_dir,
    materialize_patient_dir,
    json_to_yolo_dataset,
    run_yolo_predict,
    yolo2singlecell,
    fast_gt2singlecell,
    DataTxtGenerator,
    release_memory,
    run_fold_pipeline,
)


def build_shared_seg_split(all_patients_df: pd.DataFrame, mode: str, seg_val_ratio: float, random_state: int, seg_assignment_csv: Path | None):
    """
    共享分割模型的数据划分（独立于5-fold分类划分）
    - independent_random: 从全部患者中独立随机切分 seg_train/seg_val
    - independent_reuse: 读取已有 seg 划分 csv（列: 正式编号, seg_split）
    """
    df = all_patients_df.copy()
    if "正式编号" not in df.columns:
        raise ValueError("all_patients_df 缺少 正式编号 列")

    df["正式编号"] = df["正式编号"].astype(str).str.strip()

    if mode == "independent_reuse":
        if seg_assignment_csv is None or not seg_assignment_csv.exists():
            raise FileNotFoundError(f"未找到 seg 划分文件: {seg_assignment_csv}")
        seg_df = pd.read_csv(seg_assignment_csv)
        need_cols = {"正式编号", "seg_split"}
        if not need_cols.issubset(seg_df.columns):
            raise ValueError(f"{seg_assignment_csv} 缺少列: {need_cols}")
        seg_df["正式编号"] = seg_df["正式编号"].astype(str).str.strip()
        seg_df["seg_split"] = seg_df["seg_split"].astype(str).str.strip().str.lower()
        merged = df[["正式编号"]].drop_duplicates().merge(seg_df[["正式编号", "seg_split"]], on="正式编号", how="inner")
        train_ids = merged.loc[merged["seg_split"] == "train", "正式编号"].tolist()
        val_ids = merged.loc[merged["seg_split"] == "val", "正式编号"].tolist()
        if len(train_ids) == 0 or len(val_ids) == 0:
            raise ValueError("seg 划分结果为空，请检查 seg_assignment_csv")
        return train_ids, val_ids

    # independent_random
    uniq = df[["正式编号"]].drop_duplicates().copy().reset_index(drop=True)
    if "患者大类型_norm" in df.columns:
        type_df = df[["正式编号", "患者大类型_norm"]].drop_duplicates()
        uniq = uniq.merge(type_df, on="正式编号", how="left")
        y = uniq["患者大类型_norm"].fillna("UNK").values
    else:
        y = None

    if len(uniq) < 2:
        raise ValueError("患者数量不足，无法构建独立分割训练/验证集合")

    if y is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=seg_val_ratio, random_state=random_state)
        train_idx, val_idx = next(sss.split(uniq["正式编号"].values, y))
    else:
        val_n = max(1, int(len(uniq) * seg_val_ratio))
        train_idx = list(range(len(uniq) - val_n))
        val_idx = list(range(len(uniq) - val_n, len(uniq)))

    train_ids = uniq.iloc[train_idx]["正式编号"].tolist()
    val_ids = uniq.iloc[val_idx]["正式编号"].tolist()
    return train_ids, val_ids


def prepare_shared_segmentation_independent(args, folds: List[FoldInfo], all_patients_df: pd.DataFrame, out_root: Path) -> Dict[str, Path]:
    """
    方案三（改进版）：
    - 共享分割模型训练/验证集来自“独立 seg 划分”，不再绑定某个 fold 的 validation 集
    - 共享 singlecell 只生成一次
    """
    shared_root = out_root / "shared_segmentation_independent"
    shared_main = shared_root / "raw_main"
    ensure_clean_dir(shared_main)

    all_ids = sorted({pid for f in folds for pid in (f.train_ids + f.val_ids)})
    for pid in all_ids:
        materialize_patient_dir(Path(args.img_root) / pid, shared_main / pid)

    seg_train_ids, seg_val_ids = build_shared_seg_split(
        all_patients_df=all_patients_df,
        mode=args.seg_split_mode,
        seg_val_ratio=args.seg_val_ratio,
        random_state=args.random_state,
        seg_assignment_csv=Path(args.seg_assignment_csv) if args.seg_assignment_csv else None,
    )

    pd.DataFrame({
        "正式编号": seg_train_ids + seg_val_ids,
        "seg_split": ["train"] * len(seg_train_ids) + ["val"] * len(seg_val_ids),
    }).to_csv(shared_root / "seg_patient_split.csv", index=False, encoding="utf-8-sig")

    yolo_train_dir = shared_root / "yolo_split" / "Train"
    yolo_val_dir = shared_root / "yolo_split" / "Val"
    ensure_clean_dir(yolo_train_dir)
    ensure_clean_dir(yolo_val_dir)

    seg_train_set = set(seg_train_ids)
    seg_val_set = set(seg_val_ids)
    for pid in sorted(seg_train_set):
        if (shared_main / pid).exists():
            materialize_patient_dir(shared_main / pid, yolo_train_dir / pid)
    for pid in sorted(seg_val_set):
        if (shared_main / pid).exists():
            materialize_patient_dir(shared_main / pid, yolo_val_dir / pid)

    yolo_dataset = shared_root / "yolo_dataset"
    ensure_clean_dir(yolo_dataset)
    json_to_yolo_dataset(yolo_train_dir, yolo_val_dir, yolo_dataset)

    model = YOLO(args.yolo_init_weight)
    model.train(
        cfg=args.yolo_train_cfg,
        data=str(yolo_dataset / "dataset.yaml"),
        name=f"{args.exp_name}_shared_seg_independent",
        epochs=args.yolo_epochs,
        batch=args.yolo_batch,
        patience=args.yolo_patience,
    )
    yolo_best = Path(model.trainer.best)
    del model
    release_memory()

    pred_root = shared_root / "yolo_preds"
    data_roots = {
        "train": shared_main,
        "test_BJH": Path(args.test_bjh_root),
        "test_FXH_noALL": Path(args.test_fxh_root),
        "test_TJMU": Path(args.test_tjmu_root),
    }
    run_yolo_predict(
        yolo_best,
        data_roots,
        pred_root,
        batch=args.yolo_predict_batch,
        chunk_size=args.yolo_predict_chunk_size,
        device=args.yolo_device,
        use_half=(not args.yolo_predict_no_half),
    )

    singlecell_root = shared_root / "singlecell"
    for split_name, input_root in data_roots.items():
        yolo2singlecell(
            seg_json_dir=pred_root / split_name,
            points_json_dir=input_root,
            image_dir=input_root,
            output_base_dir=singlecell_root / split_name,
            remove_background=False,
            filter_edge_cells=True,
            min_circularity=args.min_circularity,
            min_area=args.min_area,
            crop_size=args.crop_size,
            output_size=args.output_size,
            iou_threshold=args.iou_threshold,
            num_workers=args.yolo2sc_workers,
        )

    fast_gt2singlecell(
        points_json_dir=shared_main,
        output_base_dir=singlecell_root / "train_groundtruth",
        remove_background=False,
        filter_edge_cells=True,
        min_circularity=args.min_circularity,
        min_area=args.min_area,
        crop_size=args.crop_size,
        output_size=args.output_size,
        max_workers=args.gt_workers,
    )
    DataTxtGenerator(str(singlecell_root))
    return {"shared_root": shared_root, "singlecell_root": singlecell_root}


def parse_args():
    p = argparse.ArgumentParser(description="方案三(独立分割划分版): 共享分割 + 5fold分类")
    p.add_argument("--repo-root", default="/root/autodl-tmp/projects/myq/SingleCellProject")
    p.add_argument("--exp-name", default="exp_5fold")
    p.add_argument("--excel-path", default="/root/autodl-tmp/data/patient_data_260323.xlsx")
    p.add_argument("--sheet-name", default="总表")
    p.add_argument("--patient-xlsx", default="/root/autodl-tmp/data/patient_data_260323.xlsx")
    p.add_argument("--img-root", default="/root/autodl-tmp/data/MAIN_imgs_260323")
    p.add_argument("--output-root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold")

    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--folds", nargs="*", type=int, default=None)
    p.add_argument("--fold-split-mode", choices=["random", "reuse"], default="reuse")
    p.add_argument("--fold-assignment-csv", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold/patient_base_fold_assignment.csv")

    p.add_argument("--seg-split-mode", choices=["independent_random", "independent_reuse"], default="independent_random")
    p.add_argument("--seg-val-ratio", type=float, default=0.2)
    p.add_argument("--seg-assignment-csv", default=None, help="independent_reuse 时使用，包含 正式编号,seg_split(train/val)")

    p.add_argument("--yolo-init-weight", default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/cellseg/260323_MAIN_yolo11m/weights/best.pt")
    p.add_argument("--yolo-train-cfg", default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolotrain_1.0.yaml")
    p.add_argument("--yolo-epochs", type=int, default=50)
    p.add_argument("--yolo-batch", type=int, default=32)
    p.add_argument("--yolo-patience", type=int, default=10)
    p.add_argument("--yolo-predict-batch", type=int, default=4)
    p.add_argument("--yolo-predict-chunk-size", type=int, default=256)
    p.add_argument("--yolo-device", default="0")
    p.add_argument("--yolo-predict-no-half", action="store_true")

    p.add_argument("--test-bjh-root", default="/root/autodl-tmp/data/BJH_imgs_260211")
    p.add_argument("--test-fxh-root", default="/root/autodl-tmp/data/FXH_imgs_noALL_260318")
    p.add_argument("--test-tjmu-root", default="/root/autodl-tmp/data/TJMU_imgs_260318")

    p.add_argument("--crop-size", type=int, default=576)
    p.add_argument("--output-size", type=int, default=576)
    p.add_argument("--min-circularity", type=float, default=0.65)
    p.add_argument("--min-area", type=int, default=10000)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--yolo2sc-workers", type=int, default=8)
    p.add_argument("--gt-workers", type=int, default=8)
    p.add_argument("--cls-init-ckpt", default="/root/autodl-tmp/projects/mwh/SingleCellProject/weights/pytorch_model.bin")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.fold_split_mode == "random":
        base_df, folds, _ = build_folds(
            excel_path=Path(args.excel_path),
            sheet_name=args.sheet_name,
            img_root=Path(args.img_root),
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        base_df.to_csv(out_root / "patient_base_fold_assignment.csv", index=False, encoding="utf-8-sig")
    else:
        base_df, folds, _ = load_folds_from_assignment(
            assignment_csv=Path(args.fold_assignment_csv),
            img_root=Path(args.img_root),
            n_splits=args.n_splits,
        )

    shared_artifacts = prepare_shared_segmentation_independent(args, folds, base_df, out_root)

    selected_folds = set(args.folds) if args.folds else None
    for fold in folds:
        if selected_folds and fold.fold_id not in selected_folds:
            continue
        print(f"\n{'=' * 100}\n🚀 开始执行 Fold {fold.fold_id}\n{'=' * 100}")
        run_fold_pipeline(args, fold, shared_singlecell_root=shared_artifacts["singlecell_root"])


if __name__ == "__main__":
    main()
