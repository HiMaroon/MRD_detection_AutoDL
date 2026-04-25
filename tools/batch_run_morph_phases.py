#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    desc: str
    morph_cols: List[str]


PHASES: Dict[str, PhaseSpec] = {
    "phase0": PhaseSpec("phase0", "baseline_no_morph", []),
    "phase1": PhaseSpec("phase1", "contour_3", ["area", "perimeter", "circularity"]),
    "phase2": PhaseSpec("phase2", "contour_6", ["area", "perimeter", "circularity", "aspect_ratio", "solidity", "eccentricity"]),
    "phase3": PhaseSpec("phase3", "contour_10", ["area", "perimeter", "circularity", "aspect_ratio", "extent", "convex_area", "solidity", "equiv_diameter", "major_axis_length", "eccentricity"]),
    "phase4": PhaseSpec("phase4", "contour_plus_light_appearance_14", ["area", "perimeter", "circularity", "aspect_ratio", "extent", "convex_area", "solidity", "equiv_diameter", "major_axis_length", "eccentricity", "mean_h", "mean_s", "std_s", "gray_std"]),
    "phase5": PhaseSpec("phase5", "recommended_18", ["area", "perimeter", "circularity", "aspect_ratio", "extent", "convex_area", "solidity", "equiv_diameter", "major_axis_length", "eccentricity", "mean_h", "mean_s", "std_s", "gray_std", "entropy", "texture_contrast", "laplacian_var", "dark_region_ratio"]),
    "phase6": PhaseSpec("phase6", "all_features", []),
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def write_temp_configs(repo_root: Path, data_cfg: dict, model_cfg: dict, train_cfg: dict, wandb_cfg: dict) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory(prefix="morph_cfg_")
    cfg_dir = Path(td.name) / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    init_src = (repo_root / "configs" / "__init__.py").read_text(encoding="utf-8")
    (cfg_dir / "__init__.py").write_text(init_src, encoding="utf-8")
    save_yaml(cfg_dir / "data.yaml", data_cfg)
    save_yaml(cfg_dir / "model.yaml", model_cfg)
    save_yaml(cfg_dir / "train.yaml", train_cfg)
    save_yaml(cfg_dir / "wandb.yaml", wandb_cfg)
    return td


def parse_csv(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty csv argument")
    return vals


def resolve_phase6_features() -> List[str]:
    return [
        "area", "perimeter", "circularity", "aspect_ratio", "extent", "convex_area", "solidity", "equiv_diameter",
        "major_axis_length", "minor_axis_length", "eccentricity", "mean_r", "mean_g", "mean_b", "std_r", "std_g",
        "std_b", "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v", "gray_mean", "gray_std", "entropy",
        "texture_energy", "texture_contrast", "laplacian_var", "dark_region_ratio", "low_saturation_ratio", "central_compactness"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--phases", default="phase0,phase1,phase2,phase3,phase4,phase5")
    ap.add_argument("--morph-modes", default="none,loss,fusion,loss+fusion")
    ap.add_argument("--output-root-base", required=True)
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--name-prefix", default="morph_grid")
    ap.add_argument("--python", default="python")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--restore-configs", action="store_true")
    ap.add_argument("--config-mode", choices=["temp", "inplace"], default="temp")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_yaml = repo_root / "configs" / "data.yaml"
    model_yaml = repo_root / "configs" / "model.yaml"
    train_yaml = repo_root / "configs" / "train.yaml"
    wandb_yaml = repo_root / "configs" / "wandb.yaml"

    orig_data = load_yaml(data_yaml)
    orig_model = load_yaml(model_yaml)
    orig_train = load_yaml(train_yaml)
    orig_wandb = load_yaml(wandb_yaml)

    phases = parse_csv(args.phases)
    modes = parse_csv(args.morph_modes)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")

    runs = []
    # 外循环 morph_mode，内循环 phase
    for mode in modes:
        for phase_key in phases:
            phase = PHASES[phase_key]
            cols = resolve_phase6_features() if phase_key == "phase6" else phase.morph_cols
            if phase_key == "phase0" and mode != "none":
                continue
            if not cols and mode != "none":
                continue
            run_name = f"{args.name_prefix}_{mode}_{phase.key}_{phase.desc}_{ts}".replace("+", "plus")
            runs.append((phase, cols, mode, run_name, Path(args.output_root_base) / run_name))

    failed = []
    try:
        for phase, cols, mode, run_name, out_root in runs:
            data_cfg = copy.deepcopy(orig_data)
            model_cfg = copy.deepcopy(orig_model)
            train_cfg = copy.deepcopy(orig_train)
            wandb_cfg = copy.deepcopy(orig_wandb)

            data_cfg["morph_cols"] = cols
            data_cfg["morph_dim"] = len(cols)
            data_cfg["return_morph"] = mode != "none"
            model_cfg["morph_mode"] = mode
            model_cfg["morph_dim"] = len(cols)
            model_cfg["use_morph_head"] = mode in {"loss", "fusion", "loss+fusion"}
            train_cfg["output_root"] = str(out_root)
            wandb_cfg["project"] = args.wandb_project
            wandb_cfg["name"] = run_name

            env = os.environ.copy()
            tmp = None
            if args.config_mode == "inplace":
                save_yaml(data_yaml, data_cfg)
                save_yaml(model_yaml, model_cfg)
                save_yaml(train_yaml, train_cfg)
                save_yaml(wandb_yaml, wandb_cfg)
            else:
                tmp = write_temp_configs(repo_root, data_cfg, model_cfg, train_cfg, wandb_cfg)
                old_py = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{tmp.name}:{repo_root}" + (f":{old_py}" if old_py else "")

            cmd = [args.python, "tools/train.py"]
            print(f"[RUN] mode={mode} phase={phase.key} cmd={' '.join(cmd)}")
            if not args.dry_run:
                proc = subprocess.run(cmd, cwd=repo_root, env=env)
                if proc.returncode != 0:
                    failed.append((mode, phase.key, proc.returncode))
                    if not args.continue_on_error:
                        raise RuntimeError(f"failed: {mode} {phase.key}")
            if tmp is not None:
                tmp.cleanup()
    finally:
        if args.restore_configs and args.config_mode == "inplace":
            save_yaml(data_yaml, orig_data)
            save_yaml(model_yaml, orig_model)
            save_yaml(train_yaml, orig_train)
            save_yaml(wandb_yaml, orig_wandb)

    if failed:
        print(f"[FAILED] {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
python tools/batch_run_morph_phases.py \
  --repo-root . \
  --phases phase1,phase2 \
  --morph-modes fusion,loss+fusion \
  --output-root-base /root/autodl-tmp/projects/myq/SingleCellProject/outputs/morph_grid_roi \
  --wandb-project singlecell_morph_ablation \
  --name-prefix exp_roi \
  --continue-on-error \
  --config-mode inplace
'''