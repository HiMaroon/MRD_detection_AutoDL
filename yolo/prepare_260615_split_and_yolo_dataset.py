import json
import random
import shutil
from pathlib import Path


SRC_ROOT = Path("/root/autodl-tmp/data/MAIN_imgs_260615")
SPLIT_ROOT = Path("/root/autodl-tmp/data/MAIN_imgs_split_260615")
YOLO_ROOT = Path("/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_dataset_260615")
SEED = 260615
TRAIN_RATIO = 0.7

CELL_DICT_BIG = {
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2,
}


def copy_patient_dirs(src_dirs, dst_root):
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in src_dirs:
        dst = dst_root / src.name
        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".ipynb_checkpoints"))


def split_patients():
    patients = sorted(p for p in SRC_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not patients:
        raise RuntimeError(f"No patient directories found under {SRC_ROOT}")

    rng = random.Random(SEED)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    n_train = round(len(shuffled) * TRAIN_RATIO)
    train_patients = sorted(shuffled[:n_train], key=lambda p: p.name)
    val_patients = sorted(shuffled[n_train:], key=lambda p: p.name)

    copy_patient_dirs(train_patients, SPLIT_ROOT / "Train")
    copy_patient_dirs(val_patients, SPLIT_ROOT / "Val")

    manifest = {
        "source": str(SRC_ROOT),
        "split_root": str(SPLIT_ROOT),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "num_patients": len(patients),
        "num_train_patients": len(train_patients),
        "num_val_patients": len(val_patients),
        "train_patients": [p.name for p in train_patients],
        "val_patients": [p.name for p in val_patients],
    }
    with open(SPLIT_ROOT / "split_manifest_260615.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def iter_labelme_jsons(root):
    for path in root.rglob("*.json"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        yield path


def process_json_files(json_files, image_output_dir, label_output_dir):
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_no_label = 0
    skipped_missing_image = 0
    skipped_labels = set()

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"JSON load failed: {json_file}: {exc}")
            continue

        image_name = data.get("imagePath")
        image_height = data.get("imageHeight")
        image_width = data.get("imageWidth")
        if not image_name or not image_height or not image_width:
            continue

        image_path = json_file.parent / image_name
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        yolo_lines = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue
            label = str(shape.get("label", ""))
            if label not in CELL_DICT_BIG:
                skipped_labels.add(label)
                continue
            normalized = []
            for x, y in shape.get("points", []):
                normalized.append(f"{float(x) / float(image_width):.6f}")
                normalized.append(f"{float(y) / float(image_height):.6f}")
            if len(normalized) >= 6:
                yolo_lines.append(f"0 {' '.join(normalized)}")

        if not yolo_lines:
            skipped_no_label += 1
            continue

        dst_image = image_output_dir / image_path.name
        dst_label = label_output_dir / f"{image_path.stem}.txt"
        if dst_image.exists() or dst_label.exists():
            raise FileExistsError(f"Duplicate YOLO image/label stem detected: {image_path.stem}")
        shutil.copy2(image_path, dst_image)
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        converted += 1

    return {
        "converted": converted,
        "skipped_no_label": skipped_no_label,
        "skipped_missing_image": skipped_missing_image,
        "skipped_labels": sorted(skipped_labels),
    }


def build_yolo_dataset():
    train_root = SPLIT_ROOT / "Train"
    val_root = SPLIT_ROOT / "Val"
    stats = {
        "train": process_json_files(list(iter_labelme_jsons(train_root)), YOLO_ROOT / "images" / "train", YOLO_ROOT / "labels" / "train"),
        "val": process_json_files(list(iter_labelme_jsons(val_root)), YOLO_ROOT / "images" / "val", YOLO_ROOT / "labels" / "val"),
    }

    yaml_text = f"""# Ultralytics YOLO dataset generated for MAIN 260615
path: {YOLO_ROOT}
train: images/train
val: images/val
nc: 1
names:
  0: cell
"""
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    with open(YOLO_ROOT / "source_dirs.txt", "w", encoding="utf-8") as f:
        f.write("# train\n")
        f.write(f"{train_root}\n\n# val\n{val_root}\n")
    with open(YOLO_ROOT / "conversion_stats_260615.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    if SPLIT_ROOT.exists():
        raise FileExistsError(f"Split target already exists: {SPLIT_ROOT}")
    if YOLO_ROOT.exists():
        raise FileExistsError(f"YOLO target already exists: {YOLO_ROOT}")

    manifest = split_patients()
    stats = build_yolo_dataset()
    print(json.dumps({"split": manifest, "yolo": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
