import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


if __name__ == '__main__':
    project_root = Path("/root/autodl-tmp/projects/myq/SingleCellProject")
    train_cfg = project_root / "yolo" / "yolotrain_1.0.yaml"
    dataset_yaml = project_root / "yolo" / "yolo_dataset_260615" / "dataset.yaml"
    run_name = "260615_MAIN_yolo11m"
    pred_root = project_root / "yolo" / "yolo_preds_260615"
    split_roots = {
        "train": Path("/root/autodl-tmp/data/MAIN_imgs_split_260615/Train"),
        "val": Path("/root/autodl-tmp/data/MAIN_imgs_split_260615/Val"),
    }

    model = YOLO(str(project_root / "weights" / "yolo11m-seg.pt"))
    model.train(
        cfg=str(train_cfg),
        data=str(dataset_yaml),
        project=str(project_root / "yolo" / "cellseg"),
        name=run_name,
        batch=32,
        epochs=250,
        patience=50,
        exist_ok=False,
    )

    best_model_path = project_root / "yolo" / "cellseg" / run_name / "weights" / "best.pt"
    pred_model = YOLO(str(best_model_path))

    for split_name, input_dir in split_roots.items():
        output_base = pred_root / split_name
        image_list = sorted(input_dir.rglob("*.jpg"))
        print(f"Predicting {split_name}: {len(image_list)} images")
        results = pred_model.predict(
            source=image_list,
            batch=1,
            save=False,
            verbose=True,
            conf=0.25,
            iou=0.5,
            imgsz=1280,
            stream=True,
        )
        for result in tqdm(results, desc=f"Saving {split_name}", unit="img"):
            original_path = Path(result.path)
            try:
                rel_path = original_path.relative_to(input_dir)
            except ValueError:
                rel_path = Path(original_path.name)

            new_path = output_base / rel_path
            new_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(new_path))
            result.save_txt(str(new_path.with_suffix(".txt")))

            parsed_json = json.loads(result.to_json())
            with open(new_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
