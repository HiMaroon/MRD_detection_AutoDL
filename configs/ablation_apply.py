from pathlib import Path
import argparse
import copy
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs"
ABL_DIR = CFG_DIR / "ablation"


def deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def load_yaml(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    txt = p.read_text(encoding="utf-8")
    data = yaml.safe_load(txt)
    return {} if data is None else data


def dump_yaml(p: Path, data: dict):
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def backup_file(p: Path):
    b = p.with_suffix(p.suffix + ".bak")
    b.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return b


def clean_data_top_level_duplicates(data_cfg: dict):
    """
    这些键应只存在于 data.yaml 的 advanced 下，不应出现在顶层。
    """
    dup_keys = [
        "input_mode",
        "center_weight_mode", "center_weight_strength", "center_weight_sigma",
        "border_aug_prob", "border_aug_width_ratio", "border_aug_mode",
    ]
    removed = []
    for k in dup_keys:
        if k in data_cfg:
            data_cfg.pop(k)
            removed.append(k)
    return removed


def main():
    parser = argparse.ArgumentParser(description="Apply ablation yaml safely to configs/*.yaml")
    parser.add_argument(
        "--ablation",
        # default="center_border",
        default="baseline",
        help="ablation name or path, e.g. center_border_only OR configs/ablation/center_border_only.yaml",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned changes")
    args = parser.parse_args()

    # resolve ablation path
    ablation_arg = Path(args.ablation)
    if ablation_arg.exists():
        ablation_path = ablation_arg
    else:
        name = args.ablation
        if not name.endswith(".yaml"):
            name += ".yaml"
        ablation_path = ABL_DIR / name

    ab = load_yaml(ablation_path)
    ab_adv = ab.get("advanced", {})
    ab_model = ab.get("model", {})
    ab_train = ab.get("train", {})

    data_path = CFG_DIR / "data.yaml"
    model_path = CFG_DIR / "model.yaml"
    train_path = CFG_DIR / "train.yaml"

    data_cfg = load_yaml(data_path)
    model_cfg = load_yaml(model_path)
    train_cfg = load_yaml(train_path)

    # ensure advanced exists
    if "advanced" not in data_cfg or not isinstance(data_cfg["advanced"], dict):
        data_cfg["advanced"] = {}

    # merge
    deep_update(data_cfg["advanced"], ab_adv)
    deep_update(model_cfg, ab_model)
    deep_update(train_cfg, ab_train)

    removed_keys = clean_data_top_level_duplicates(data_cfg)

    print(f"[INFO] ablation: {ablation_path}")
    print(f"[INFO] patch advanced keys: {list(ab_adv.keys())}")
    print(f"[INFO] patch model keys: {list(ab_model.keys())}")
    print(f"[INFO] patch train keys: {list(ab_train.keys())}")
    if removed_keys:
        print(f"[INFO] removed duplicated top-level keys in data.yaml: {removed_keys}")

    if args.dry_run:
        print("[DRY-RUN] no file written.")
        return

    if not args.no_backup:
        for p in [data_path, model_path, train_path]:
            b = backup_file(p)
            print(f"[BACKUP] {b}")

    dump_yaml(data_path, data_cfg)
    dump_yaml(model_path, model_cfg)
    dump_yaml(train_path, train_cfg)
    print("[OK] Applied ablation successfully.")


if __name__ == "__main__":
    main()
