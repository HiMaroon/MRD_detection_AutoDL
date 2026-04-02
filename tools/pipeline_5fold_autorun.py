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
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ultralytics import YOLO
import torch

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


def parse_patient_id_from_image_name(name: str) -> str | None:
    stem = Path(name).stem
    prefix = stem.split("_")[0]
    parts = prefix.split("-")
    if len(parts) < 2:
        return None
    return f"{parts[0]}-{parts[1]}"


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
        # 非 Linux 或无 libc.so.6 时忽略
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


def fast_gt2singlecell(points_json_dir: Path, output_base_dir: Path, *, remove_background: bool,
                       filter_edge_cells: bool, min_circularity: float, min_area: int,
                       crop_size: int, output_size: int, max_workers: int = 8):
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
        list(tqdm(executor.map(_run_one, json_files), total=len(json_files), desc=f"fast_gt2singlecell:{points_json_dir.name}"))


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
# -----------------------------
def build_folds(excel_path: Path, sheet_name: str, img_root: Path, n_splits: int, random_state: int) -> Tuple[pd.DataFrame, List[FoldInfo], pd.DataFrame]:
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


def load_folds_from_assignment(assignment_csv: Path, img_root: Path, n_splits: int) -> Tuple[pd.DataFrame, List[FoldInfo], pd.DataFrame]:
    if not assignment_csv.exists():
        raise FileNotFoundError(f"未找到 fold 划分文件: {assignment_csv}")

    df = pd.read_csv(assignment_csv)
    required_cols = {"正式编号", "fold"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"{assignment_csv} 缺少必要列: {required_cols}")

    df["正式编号"] = df["正式编号"].astype(str).str.strip()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df[df["fold"].notna()].copy()
    df["fold"] = df["fold"].astype(int)

    df["folder_exists"] = df["正式编号"].apply(lambda x: (img_root / str(x).strip()).is_dir())
    missing_df = df[~df["folder_exists"]].copy()
    df = df[df["folder_exists"]].copy().reset_index(drop=True)

    available_folds = sorted(df["fold"].unique().tolist())
    if len(available_folds) < n_splits:
        print(f"⚠️ assignment 中仅检测到 {len(available_folds)} 个 fold，参数 n_splits={n_splits}")

    folds: List[FoldInfo] = []
    for fold_id in available_folds:
        val_ids = df.loc[df["fold"] == fold_id, "正式编号"].tolist()
        train_ids = df.loc[df["fold"] != fold_id, "正式编号"].tolist()
        folds.append(FoldInfo(fold_id=fold_id, train_ids=train_ids, val_ids=val_ids))
    return df, folds, missing_df


# -----------------------------
# 2) YOLO 数据转换 + 训练 + 推理
# -----------------------------
def collect_jsons(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.json") if ".ipynb_checkpoints" not in str(p)]


def json_to_yolo_dataset(train_root: Path, val_root: Path, output_dir: Path):
    img_train = output_dir / "images" / "train"
    img_val = output_dir / "images" / "val"
    lbl_train = output_dir / "labels" / "train"
    lbl_val = output_dir / "labels" / "val"
    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    class_mapping = {"0": 0}

    def process(json_files: List[Path], out_img: Path, out_lbl: Path):
        for js in tqdm(json_files, desc=f"json2yolo->{out_img.parent.name}/{out_img.name}"):
            data = json.loads(js.read_text(encoding="utf-8"))
            image_name = data.get("imagePath")
            h, w = data.get("imageHeight"), data.get("imageWidth")
            if not image_name or not h or not w:
                continue
            src_img = js.parent / image_name
            if not src_img.exists():
                continue
            shutil.copy2(src_img, out_img / image_name)

            lines = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue
                cls = class_mapping.get(str(shape.get("label", "0")), 0)
                points = shape.get("points", [])
                coords = []
                for x, y in points:
                    coords.extend([f"{x / w:.6f}", f"{y / h:.6f}"])
                if coords:
                    lines.append(f"{cls} {' '.join(coords)}")
            (out_lbl / f"{Path(image_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    process(collect_jsons(train_root), img_train, lbl_train)
    process(collect_jsons(val_root), img_val, lbl_val)

    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "0"},
    }
    with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)


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
            continue
        output_base = pred_root / split_name
        # 关键：分块推理，避免将超大 source 列表一次性交给 ultralytics 导致内存飙升
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
                        # batch=1 仍 OOM，回退到逐图推理保证流程能继续
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
                # 分块结束后及时回收本块内存
                if results is not None:
                    del results
                del chunk_paths
                release_memory()
    # 显式释放 YOLO 推理模型显存，避免影响后续分类训练
    del model
    release_memory()


# -----------------------------
# 3) 分类训练（两阶段）
# -----------------------------
def patch_train_configs(repo_root: Path, *, train_labels: Path, val_labels: Path, output_root: Path, local_weight_path: Path | None):
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
        # Stage1 必须从默认初始化 ckpt 起跑（而不是上一折/上一阶段残留 ckpt）
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
        run_test_on_split(split="val", ckpt_path=str(final_ckpt), test_data_sir=str(label_path), output_dir=str(out_dir))

    set_chinese_font()
    for split_name in labels_map:
        run_one_task(
            cell_result_csv=str(fold_root / "eval" / split_name / "val_results.csv"),
            patient_info_xlsx=str(patient_xlsx),
            output_png=str(fold_root / "patient_report" / split_name / "patient_ratio_from_cell_results.png"),
            output_excel=str(fold_root / "patient_report" / split_name / "patient_ratio_from_cell_results.xlsx"),
        )


# -----------------------------
# 5) 共享分割（方案三）
# -----------------------------
def filter_labels_by_patient_ids(src_txt: Path, dst_txt: Path, patient_ids: set[str]):
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for line in src_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        img_path = parts[0]
        pid = parse_patient_id_from_image_name(Path(img_path).name)
        if pid in patient_ids:
            kept.append(line)
    dst_txt.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def prepare_shared_segmentation(args, folds: List[FoldInfo], out_root: Path) -> Dict[str, Path]:
    """
    方案三核心：
    - 训练一套共享 YOLO 分割模型
    - 只生成一套共享 singlecell
    - 后续各 fold 只在分类阶段切 train/val 标签
    """
    shared_root = out_root / "shared_segmentation"
    shared_main = shared_root / "raw_main"
    ensure_clean_dir(shared_main)

    all_ids = sorted({pid for f in folds for pid in (f.train_ids + f.val_ids)})
    for pid in all_ids:
        materialize_patient_dir(Path(args.img_root) / pid, shared_main / pid)

    yolo_dataset = shared_root / "yolo_dataset"
    ensure_clean_dir(yolo_dataset)
    seg_val_ids = set(folds[0].val_ids)
    seg_train_ids = [pid for pid in all_ids if pid not in seg_val_ids]
    yolo_train_dir = shared_root / "yolo_split" / "Train"
    yolo_val_dir = shared_root / "yolo_split" / "Val"
    ensure_clean_dir(yolo_train_dir)
    ensure_clean_dir(yolo_val_dir)
    for pid in seg_train_ids:
        materialize_patient_dir(shared_main / pid, yolo_train_dir / pid)
    for pid in seg_val_ids:
        materialize_patient_dir(shared_main / pid, yolo_val_dir / pid)

    json_to_yolo_dataset(yolo_train_dir, yolo_val_dir, yolo_dataset)

    yolo_run_name = f"{args.exp_name}_shared_seg"
    model = YOLO(args.yolo_init_weight)
    model.train(
        cfg=args.yolo_train_cfg,
        data=str(yolo_dataset / "dataset.yaml"),
        name=yolo_run_name,
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


# -----------------------------
# Pipeline
# -----------------------------
def run_fold_pipeline(args, fold: FoldInfo, shared_singlecell_root: Path):
    fold_root = Path(args.output_root) / f"fold{fold.fold_id}"
    singlecell_root = fold_root / "singlecell"
    ensure_clean_dir(singlecell_root)

    train_ids = set(fold.train_ids)
    val_ids = set(fold.val_ids)

    filter_labels_by_patient_ids(shared_singlecell_root / "train_labels.txt", singlecell_root / "train_labels.txt", train_ids)
    filter_labels_by_patient_ids(shared_singlecell_root / "train_labels.txt", singlecell_root / "val_labels.txt", val_ids)
    filter_labels_by_patient_ids(shared_singlecell_root / "train_groundtruth_labels.txt", singlecell_root / "train_groundtruth_labels.txt", train_ids)

    for ext_name in ["test_BJH", "test_FXH_noALL", "test_TJMU"]:
        src = shared_singlecell_root / f"{ext_name}_labels.txt"
        dst = singlecell_root / f"{ext_name}_labels.txt"
        if src.exists():
            shutil.copy2(src, dst)

    # 仅分类阶段 5-fold 训练（共享 singlecell）
    final_ckpt = run_two_stage_classifier_train(
        repo_root=Path(args.repo_root),
        fold_root=fold_root,
        train_labels_gt=singlecell_root / "train_groundtruth_labels.txt",
        train_labels_yolo=singlecell_root / "train_labels.txt",
        val_labels_yolo=singlecell_root / "val_labels.txt",
        cls_init_ckpt=Path(args.cls_init_ckpt) if args.cls_init_ckpt else None,
    )

    # 8,9) 测试 + 患者级分析
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
    p = argparse.ArgumentParser(description="5-fold 自动化训练流水线（方案三：共享分割 + 5折分类融合）")
    p.add_argument("--repo-root", default="/root/autodl-tmp/projects/myq/SingleCellProject")
    p.add_argument("--exp-name", default="exp_5fold")
    p.add_argument("--excel-path", default="/root/autodl-tmp/data/patient_data_260323.xlsx")
    p.add_argument("--sheet-name", default="总表")
    p.add_argument("--patient-xlsx", default="/root/autodl-tmp/data/patient_data_260323.xlsx")
    p.add_argument("--img-root", default="/root/autodl-tmp/data/MAIN_imgs_260323")
    p.add_argument("--output-root", default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold")

    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--folds", nargs="*", type=int, default=None, help="仅运行指定 fold，例如 --folds 1 3")
    p.add_argument("--fold-split-mode", choices=["random", "reuse"], default="reuse", help="患者分折来源：随机重划分 或 读取已有划分")
    p.add_argument(
        "--fold-assignment-csv",
        default="/root/autodl-tmp/projects/myq/SingleCellProject/runs_5fold/patient_base_fold_assignment.csv",
        help="reuse 模式下使用的历史分折 CSV",
    )

    p.add_argument("--yolo-init-weight", default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/cellseg/260323_MAIN_yolo11m/weights/best.pt")
    p.add_argument("--yolo-train-cfg", default="/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolotrain_1.0.yaml")
    p.add_argument("--yolo-epochs", type=int, default=50)
    p.add_argument("--yolo-batch", type=int, default=32)
    p.add_argument("--yolo-patience", type=int, default=10)
    p.add_argument("--yolo-predict-batch", type=int, default=4, help="YOLO 推理 batch size")
    p.add_argument("--yolo-predict-chunk-size", type=int, default=256, help="YOLO 推理分块大小，防止内存爆炸")
    p.add_argument("--yolo-device", default="0", help="YOLO 推理设备，如 0/cpu")
    p.add_argument("--yolo-predict-no-half", action="store_true", help="禁用半精度推理（默认启用）")

    p.add_argument("--test-bjh-root", default="/root/autodl-tmp/data/BJH_imgs_260211")
    p.add_argument("--test-fxh-root", default="/root/autodl-tmp/data/FXH_imgs_noALL_260318")
    p.add_argument("--test-tjmu-root", default="/root/autodl-tmp/data/TJMU_imgs_260318")

    p.add_argument("--crop-size", type=int, default=576)
    p.add_argument("--output-size", type=int, default=576)
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

    if args.fold_split_mode == "random":
        base_df, folds, missing_df = build_folds(
            excel_path=Path(args.excel_path),
            sheet_name=args.sheet_name,
            img_root=Path(args.img_root),
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        base_df.to_csv(out_root / "patient_base_fold_assignment.csv", index=False, encoding="utf-8-sig")
    else:
        base_df, folds, missing_df = load_folds_from_assignment(
            assignment_csv=Path(args.fold_assignment_csv),
            img_root=Path(args.img_root),
            n_splits=args.n_splits,
        )
        base_df.to_csv(out_root / "patient_base_fold_assignment_reused.csv", index=False, encoding="utf-8-sig")

    if len(missing_df) > 0:
        missing_df.to_csv(out_root / "patients_missing_folders.csv", index=False, encoding="utf-8-sig")

    shared_artifacts = prepare_shared_segmentation(args, folds, out_root)

    selected_folds = set(args.folds) if args.folds else None
    for fold in folds:
        if selected_folds and fold.fold_id not in selected_folds:
            continue
        print(f"\n{'=' * 100}\n🚀 开始执行 Fold {fold.fold_id}\n{'=' * 100}")
        run_fold_pipeline(args, fold, shared_singlecell_root=shared_artifacts["singlecell_root"])


if __name__ == "__main__":
    main()
