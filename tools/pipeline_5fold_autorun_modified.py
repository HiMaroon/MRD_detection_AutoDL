import argparse
import concurrent.futures
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ultralytics import YOLO
import torch
import yaml

# 复用已有工具函数
from dataset.DataTextGenerator_test import DataTxtGenerator
from dataset.groundtruth2singlecell import process_cells_from_ground_truth
from tools.test import run_test_on_split
from tools.patient_analysis import run_one_task, set_chinese_font
from yolo.yolo2singlecell import batch_process_directory as yolo2singlecell

CENTER_PREFIXES = ("PKUPH", "BEPH", "TAB")


@dataclass
class FoldInfo:
    fold_id: int
    train_ids: List[str]
    val_ids: List[str]


# -----------------------------
# 基础工具
# -----------------------------
def normalize_patient_type(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if x in ["HC", "HD", "NORMAL", "HEALTHY"]:
        return "HC"
    if x == "AML":
        return "AML"
    return x


def get_center(patient_id: str):
    if pd.isna(patient_id):
        return None
    patient_id = str(patient_id).strip()
    for prefix in CENTER_PREFIXES:
        if patient_id.startswith(prefix):
            return prefix
    return None


def run_cmd(cmd: List[str], cwd: Path | None = None):
    print(f"\n[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def release_memory():
    """
    同时释放 GPU cache 与 Python/系统堆内存（尽力而为）。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file_with_hardlink_fallback(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_patient_dir(src: Path, dst: Path):
    """
    将患者目录“实体化”到目标目录，优先使用硬链接，避免目录软链接带来的兼容性问题。
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        _copy_file_with_hardlink_fallback(path, out)


def fast_gt2singlecell(
    points_json_dir: Path,
    output_base_dir: Path,
    *,
    remove_background: bool,
    filter_edge_cells: bool,
    min_circularity: float,
    min_area: int,
    crop_size: int,
    output_size: int,
    max_workers: int = 8,
):
    """
    针对本项目目录结构优化的 GT 单细胞提取：
    - image 与 json 位于同目录且同 stem
    - 跳过 groundtruth2singlecell.py 中昂贵的全目录递归回退搜索
    - 使用线程并行加速 I/O + OpenCV 处理
    """
    points_json_dir = Path(points_json_dir)
    output_base_dir = Path(output_base_dir)
    json_files = [p for p in points_json_dir.rglob("*.json") if ".ipynb_checkpoints" not in str(p)]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    def _run_one(js: Path):
        base = js.with_suffix("")
        image_path = None
        for ext in exts:
            candidate = base.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            return 0

        out_dir = output_base_dir / js.relative_to(points_json_dir).parent
        process_cells_from_ground_truth(
            points_json_path=js,
            image_path=image_path,
            output_dir=out_dir,
            remove_background=remove_background,
            filter_edge_cells=filter_edge_cells,
            min_circularity=min_circularity,
            min_area=min_area,
            crop_size=crop_size,
            output_size=output_size,
        )
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(_run_one, json_files),
                total=len(json_files),
                desc=f"fast_gt2singlecell:{points_json_dir.name}",
            )
        )


def copy_with_backup(path: Path) -> Tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".bak_5fold")
    content = path.read_text(encoding="utf-8")
    backup.write_text(content, encoding="utf-8")
    return content, backup


def restore_backup(path: Path, backup: Path):
    if backup.exists():
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        backup.unlink()


def find_best_ckpt(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"未在 {output_dir} 找到 ckpt 文件")
    ranked = [p for p in candidates if p.name != "last.ckpt"]
    return ranked[0] if ranked else candidates[0]


# -----------------------------
# 1) 患者 5-fold 划分
# 支持：
#   A. 从 Excel 新建分层 5-fold
#   B. 从已有 patient_base_fold_assignment.csv 直接读取
# -----------------------------
def _prepare_base_df(excel_path: Path, sheet_name: str, img_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df["正式编号"] = df["正式编号"].astype(str).str.strip()
    df["患者大类型_norm"] = df["患者大类型"].apply(normalize_patient_type)
    df["center"] = df["正式编号"].apply(get_center)

    df = df[df["center"].isin(CENTER_PREFIXES)].copy()
    df = df[df["正式编号"].notna() & (df["正式编号"] != "")].copy()
    df = df[df["正式编号"] != "未使用"].copy()
    df = df[df["患者大类型_norm"].isin(["HC", "AML"])].copy()
    df = df.drop_duplicates(subset=["正式编号"]).copy()

    df["folder_exists"] = df["正式编号"].apply(lambda x: (img_root / str(x).strip()).is_dir())
    missing_df = df[~df["folder_exists"]].copy()
    df = df[df["folder_exists"]].copy().reset_index(drop=True)
    return df, missing_df


def build_folds_from_excel(
    excel_path: Path,
    sheet_name: str,
    img_root: Path,
    n_splits: int,
    random_state: int,
) -> Tuple[pd.DataFrame, List[FoldInfo], pd.DataFrame]:
    df, missing_df = _prepare_base_df(excel_path, sheet_name, img_root)

    y = df["患者大类型_norm"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    df["fold"] = -1
    folds: List[FoldInfo] = []
    for fold_id, (_, val_idx) in enumerate(skf.split(df["正式编号"].values, y), start=1):
        df.loc[val_idx, "fold"] = fold_id
        val_ids = df.loc[val_idx, "正式编号"].tolist()
        train_ids = df.loc[~df.index.isin(val_idx), "正式编号"].tolist()
        folds.append(FoldInfo(fold_id=fold_id, train_ids=train_ids, val_ids=val_ids))

    return df, folds, missing_df


def build_folds_from_assignment_csv(
    assignment_csv: Path,
    excel_path: Path,
    sheet_name: str,
    img_root: Path,
) -> Tuple[pd.DataFrame, List[FoldInfo], pd.DataFrame]:
    """
    从已有 patient_base_fold_assignment.csv 读取 fold 分配。
    要求至少包含：
      - 正式编号
      - fold
    其余列若缺失，则用 Excel 信息补齐。
    """
    if not assignment_csv.exists():
        raise FileNotFoundError(f"未找到 fold assignment 文件: {assignment_csv}")

    assign_df = pd.read_csv(assignment_csv)
    if "正式编号" not in assign_df.columns or "fold" not in assign_df.columns:
        raise ValueError("assignment csv 至少需要包含列：'正式编号' 和 'fold'")

    assign_df["正式编号"] = assign_df["正式编号"].astype(str).str.strip()
    assign_df["fold"] = pd.to_numeric(assign_df["fold"], errors="coerce").astype("Int64")

    base_df, missing_df = _prepare_base_df(excel_path, sheet_name, img_root)

    # 用 Excel 清洗后的合法患者列表做约束，再用 assignment 的 fold 覆盖
    df = base_df.merge(
        assign_df[["正式编号", "fold"]].drop_duplicates(subset=["正式编号"]),
        on="正式编号",
        how="inner",
    ).copy()

    if df["fold"].isna().any():
        bad_ids = df.loc[df["fold"].isna(), "正式编号"].tolist()
        raise ValueError(f"以下患者 fold 为空：{bad_ids}")

    df["fold"] = df["fold"].astype(int)
    unique_folds = sorted(df["fold"].unique().tolist())

    folds: List[FoldInfo] = []
    for fold_id in unique_folds:
        val_ids = df.loc[df["fold"] == fold_id, "正式编号"].tolist()
        train_ids = df.loc[df["fold"] != fold_id, "正式编号"].tolist()
        if len(val_ids) == 0:
            raise ValueError(f"fold={fold_id} 没有验证患者")
        folds.append(FoldInfo(fold_id=fold_id, train_ids=train_ids, val_ids=val_ids))

    return df, folds, missing_df


def build_folds(
    excel_path: Path,
    sheet_name: str,
    img_root: Path,
    n_splits: int,
    random_state: int,
    assignment_csv: Path | None = None,
) -> Tuple[pd.DataFrame, List[FoldInfo], pd.DataFrame]:
    if assignment_csv is not None:
        print(f"✅ 使用已有 fold 分配文件: {assignment_csv}")
        return build_folds_from_assignment_csv(
            assignment_csv=assignment_csv,
            excel_path=excel_path,
            sheet_name=sheet_name,
            img_root=img_root,
        )

    print("✅ 未提供已有 fold 分配文件，按当前参数重新分层划分。")
    return build_folds_from_excel(
        excel_path=excel_path,
        sheet_name=sheet_name,
        img_root=img_root,
        n_splits=n_splits,
        random_state=random_state,
    )


# -----------------------------
# 2) YOLO 推理（固定单一模型，不再每个 fold 重新训练）
# -----------------------------
def run_yolo_predict(
    model_path: Path,
    data_roots: Dict[str, Path],
    pred_root: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1280,
    batch: int = 16,
    chunk_size: int = 256,
    device: str = "0",
    use_half: bool = True,
):
    model = YOLO(str(model_path))

    def _is_oom_error(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    for split_name, input_root in data_roots.items():
        images = list(input_root.rglob("*.jpg")) + list(input_root.rglob("*.png"))
        if len(images) == 0:
            print(f"⚠️ {split_name} 未找到图片，跳过。")
            continue

        output_base = pred_root / split_name
        with tqdm(total=len(images), desc=f"yolo_predict:{split_name}") as pbar:
            for start in range(0, len(images), chunk_size):
                chunk_paths = images[start:start + chunk_size]
                current_batch = min(batch, len(chunk_paths))
                results = None

                while True:
                    try:
                        results = model.predict(
                            source=[str(p) for p in chunk_paths],
                            batch=current_batch,
                            save=False,
                            verbose=False,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            stream=True,
                            device=device,
                            workers=0,
                            half=(use_half and str(device).lower() != "cpu"),
                        )
                        for result in results:
                            img_path = Path(result.path)
                            rel = img_path.relative_to(input_root)
                            out_img = output_base / rel
                            out_img.parent.mkdir(parents=True, exist_ok=True)
                            result.save(filename=str(out_img))
                            result.save_txt(str(out_img.with_suffix(".txt")))
                            with open(out_img.with_suffix(".json"), "w", encoding="utf-8") as f:
                                json.dump(json.loads(result.to_json()), f, ensure_ascii=False, indent=2)
                            pbar.update(1)
                        break

                    except RuntimeError as e:
                        if not _is_oom_error(e):
                            raise
                        release_memory()

                        if current_batch > 1:
                            current_batch = max(1, current_batch // 2)
                            print(f"⚠️ yolo_predict OOM，自动降 batch 到 {current_batch} 后重试。")
                            continue

                        print("⚠️ batch=1 仍 OOM，回退到逐图推理。")
                        for one_path in chunk_paths:
                            one_result = model.predict(
                                source=str(one_path),
                                batch=1,
                                save=False,
                                verbose=False,
                                conf=conf,
                                iou=iou,
                                imgsz=imgsz,
                                device=device,
                                workers=0,
                                half=(use_half and str(device).lower() != "cpu"),
                            )[0]
                            rel = one_path.relative_to(input_root)
                            out_img = output_base / rel
                            out_img.parent.mkdir(parents=True, exist_ok=True)
                            one_result.save(filename=str(out_img))
                            one_result.save_txt(str(out_img.with_suffix(".txt")))
                            with open(out_img.with_suffix(".json"), "w", encoding="utf-8") as f:
                                json.dump(json.loads(one_result.to_json()), f, ensure_ascii=False, indent=2)
                            pbar.update(1)
                        break

                if results is not None:
                    del results
                del chunk_paths
                release_memory()

    del model
    release_memory()


# -----------------------------
# 3) 分类训练（两阶段）
# -----------------------------
def patch_train_configs(
    repo_root: Path,
    *,
    train_labels: Path,
    val_labels: Path,
    output_root: Path,
    local_weight_path: Path | None,
):
    data_yaml = repo_root / "configs" / "data.yaml"
    train_yaml = repo_root / "configs" / "train.yaml"
    model_yaml = repo_root / "configs" / "model.yaml"

    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(train_yaml.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))

    data_cfg["train_labels"] = str(train_labels)
    data_cfg["val_labels"] = str(val_labels)
    train_cfg["output_root"] = str(output_root)
    if local_weight_path:
        model_cfg["local_weight_path"] = str(local_weight_path)

    data_yaml.write_text(yaml.safe_dump(data_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    train_yaml.write_text(yaml.safe_dump(train_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    model_yaml.write_text(yaml.safe_dump(model_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_two_stage_classifier_train(
    repo_root: Path,
    fold_root: Path,
    train_labels_gt: Path,
    train_labels_yolo: Path,
    val_labels_yolo: Path,
    cls_init_ckpt: Path | None = None,
) -> Path:
    data_yaml = repo_root / "configs" / "data.yaml"
    train_yaml = repo_root / "configs" / "train.yaml"
    model_yaml = repo_root / "configs" / "model.yaml"

    _, data_bak = copy_with_backup(data_yaml)
    _, train_bak = copy_with_backup(train_yaml)
    _, model_bak = copy_with_backup(model_yaml)

    try:
        stage1_out = fold_root / "cls_stage1_gt2yolo"
        stage1_out.mkdir(parents=True, exist_ok=True)
        patch_train_configs(
            repo_root,
            train_labels=train_labels_gt,
            val_labels=val_labels_yolo,
            output_root=stage1_out,
            local_weight_path=cls_init_ckpt,
        )
        run_cmd([sys.executable, "tools/train.py"], cwd=repo_root)
        stage1_ckpt = find_best_ckpt(stage1_out)

        stage2_out = fold_root / "cls_stage2_yolo2yolo"
        stage2_out.mkdir(parents=True, exist_ok=True)
        patch_train_configs(
            repo_root,
            train_labels=train_labels_yolo,
            val_labels=val_labels_yolo,
            output_root=stage2_out,
            local_weight_path=stage1_ckpt,
        )
        run_cmd([sys.executable, "tools/train.py"], cwd=repo_root)
        stage2_ckpt = find_best_ckpt(stage2_out)

        return stage2_ckpt
    finally:
        restore_backup(data_yaml, data_bak)
        restore_backup(train_yaml, train_bak)
        restore_backup(model_yaml, model_bak)


# -----------------------------
# 4) 报告
# -----------------------------
def run_eval_and_patient_report(
    final_ckpt: Path,
    patient_xlsx: Path,
    fold_root: Path,
    labels_map: Dict[str, Path],
):
    for split_name, label_path in labels_map.items():
        out_dir = fold_root / "eval" / split_name
        run_test_on_split(
            split="val",
            ckpt_path=str(final_ckpt),
            test_data_sir=str(label_path),
            output_dir=str(out_dir),
        )

    set_chinese_font()
    for split_name in labels_map:
        run_one_task(
            cell_result_csv=str(fold_root / "eval" / split_name / "val_results.csv"),
            patient_info_xlsx=str(patient_xlsx),
            output_png=str(fold_root / "patient_report" / split_name / "patient_ratio_from_cell_results.png"),
            output_excel=str(fold_root / "patient_report" / split_name / "patient_ratio_from_cell_results.xlsx"),
        )


# -----------------------------
# Pipeline
# 仅识别模型做 5-fold 训练
# YOLO 仅用同一套固定权重做分割预测
# -----------------------------
def run_fold_pipeline(args, fold: FoldInfo):
    fold_root = Path(args.output_root) / f"fold{fold.fold_id}"
    raw_split_root = fold_root / "raw_split"
    train_dir = raw_split_root / "Train"
    val_dir = raw_split_root / "Val"
    ensure_clean_dir(train_dir)
    ensure_clean_dir(val_dir)

    # 建立训练/验证目录（患者级目录）
    for pid in fold.train_ids:
        materialize_patient_dir(Path(args.img_root) / pid, train_dir / pid)
    for pid in fold.val_ids:
        materialize_patient_dir(Path(args.img_root) / pid, val_dir / pid)

    # 1) 固定 YOLO 权重直接推理（不再进行每 fold 的 YOLO 训练）
    pred_root = fold_root / "yolo_preds"
    data_roots = {
        "train": train_dir,
        "val": val_dir,
        "test_BJH": Path(args.test_bjh_root),
        "test_FXH_noALL": Path(args.test_fxh_root),
        "test_TJMU": Path(args.test_tjmu_root),
    }
    run_yolo_predict(
        model_path=Path(args.yolo_init_weight),
        data_roots=data_roots,
        pred_root=pred_root,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        batch=args.yolo_predict_batch,
        chunk_size=args.yolo_predict_chunk_size,
        device=args.yolo_device,
        use_half=(not args.yolo_predict_no_half),
    )

    # 2) YOLO 预测结果切单细胞
    singlecell_root = fold_root / "singlecell"
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

    # 3) GT 切单细胞（仅 train）
    fast_gt2singlecell(
        points_json_dir=train_dir,
        output_base_dir=singlecell_root / "train_groundtruth",
        remove_background=False,
        filter_edge_cells=True,
        min_circularity=args.min_circularity,
        min_area=args.min_area,
        crop_size=args.crop_size,
        output_size=args.output_size,
        max_workers=args.gt_workers,
    )

    # 4) 生成 labels txt
    DataTxtGenerator(str(singlecell_root))

    # 5) 两阶段分类训练（只有分类器做 5-fold）
    final_ckpt = run_two_stage_classifier_train(
        repo_root=Path(args.repo_root),
        fold_root=fold_root,
        train_labels_gt=singlecell_root / "train_groundtruth_labels.txt",
        train_labels_yolo=singlecell_root / "train_labels.txt",
        val_labels_yolo=singlecell_root / "val_labels.txt",
        cls_init_ckpt=Path(args.cls_init_ckpt) if args.cls_init_ckpt else None,
    )

    # 6) 测试 + 患者级分析
    run_eval_and_patient_report(
        final_ckpt=final_ckpt,
        patient_xlsx=Path(args.patient_xlsx),
        fold_root=fold_root,
        labels_map={
            "train": singlecell_root / "train_labels.txt",
            "val": singlecell_root / "val_labels.txt",
            "test_BJH": singlecell_root / "test_BJH_labels.txt",
            "test_FXH_noALL": singlecell_root / "test_FXH_noALL_labels.txt",
            "test_TJMU": singlecell_root / "test_TJMU_labels.txt",
        },
    )


def parse_args():
    p = argparse.ArgumentParser(description="5-fold 自动化训练流水线（固定 YOLO 分割 + 分类器 5-fold 训练）")
    p.add_argument("--repo-root", default="/root/autodl-tmp/projects/myq/SingleCellProject")
    p.add_argument("--exp-name", default="exp_5fold_center_border_aug10")
    p.add_argument("--excel-path", default="/root/autodl-tmp/data/patient_data_260416.xlsx")
    p.add_argument("--sheet-name", default="总表")
    p.add_argument("--patient-xlsx", default="/root/autodl-tmp/data/patient_data_260416.xlsx")
    p.add_argument("--img-root", default="/root/autodl-tmp/data/MAIN_imgs_260323")
    p.add_argument("--output-root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold_exp_5fold_center_border_aug10")

    # 新增：可直接使用已有的 patient_base_fold_assignment.csv
    p.add_argument(
        "--assignment-csv",
        default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold_260403/patient_base_fold_assignment.csv",
        help="可选。已有的 patient_base_fold_assignment.csv 路径；提供后将直接按该文件中的 fold 分配执行。",
    )

    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--folds", nargs="*", type=int, default=None, help="仅运行指定 fold，例如 --folds 1 3")

    # YOLO 仅做固定模型推理
    p.add_argument(
        "--yolo-init-weight",
        default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/cellseg/exp_5fold_260424_shared_seg_independent2/weights/best.pt",
        help="固定使用的 YOLO 分割权重，不再在每个 fold 重新训练。",
    )
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--yolo-imgsz", type=int, default=1280)
    p.add_argument("--yolo-predict-batch", type=int, default=4, help="YOLO 推理 batch size")
    p.add_argument("--yolo-predict-chunk-size", type=int, default=256, help="YOLO 推理分块大小，防止内存爆炸")
    p.add_argument("--yolo-device", default="0", help="YOLO 推理设备，如 0/cpu")
    p.add_argument("--yolo-predict-no-half", action="store_true", help="禁用半精度推理（默认启用）")

    p.add_argument("--test-bjh-root", default="/root/autodl-tmp/data/BJH_imgs_260211")
    p.add_argument("--test-fxh-root", default="/root/autodl-tmp/data/FXH_imgs_noALL_260318")
    p.add_argument("--test-tjmu-root", default="/root/autodl-tmp/data/TJMU_imgs_260416")

    p.add_argument("--crop-size", type=int, default=576)
    p.add_argument("--output-size", type=int, default=224)
    p.add_argument("--min-circularity", type=float, default=0.65)
    p.add_argument("--min-area", type=int, default=10000)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--yolo2sc-workers", type=int, default=8, help="yolo2singlecell 并行线程数")
    p.add_argument("--gt-workers", type=int, default=8, help="GT 单细胞提取并行线程数")
    p.add_argument(
        "--cls-init-ckpt",
        default="/root/autodl-tmp/projects/mwh/SingleCellProject/weights/pytorch_model.bin",
        help="分类训练 Stage1 的默认初始化 ckpt（为空则沿用 model.yaml）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_df, folds, missing_df = build_folds(
        excel_path=Path(args.excel_path),
        sheet_name=args.sheet_name,
        img_root=Path(args.img_root),
        n_splits=args.n_splits,
        random_state=args.random_state,
        assignment_csv=Path(args.assignment_csv) if args.assignment_csv else None,
    )

    # 无论是新划分还是读取旧划分，都统一导出一份当前实际使用的分配表，便于复现
    base_df.to_csv(out_root / "patient_base_fold_assignment.csv", index=False, encoding="utf-8-sig")
    if len(missing_df) > 0:
        missing_df.to_csv(out_root / "patients_missing_folders.csv", index=False, encoding="utf-8-sig")

    selected_folds = set(args.folds) if args.folds else None
    for fold in folds:
        if selected_folds and fold.fold_id not in selected_folds:
            continue
        print(f"\n{'=' * 100}\n🚀 开始执行 Fold {fold.fold_id}\n{'=' * 100}")
        run_fold_pipeline(args, fold)


if __name__ == "__main__":
    main()
