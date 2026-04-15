import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import pytorch_lightning as pl
from tqdm import tqdm

# 项目根目录加入 sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.lit_module import LitSingleCell
from src.augmentations import CenterWeightTransform, BorderSuppressionAugment, MacenkoNormalizer, HEDeconvolution
from configs import data_cfg, model_cfg


class _Wrapper(pl.LightningModule):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        return self.core(x)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = self.target_layer.register_forward_hook(self._save_activation)
        self.h2 = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook 未正确捕获到 activation 或 gradient。")

        # Grad-CAM: 对梯度做 GAP 得权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        # 归一化到 [0, 1]
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, logits

    def close(self):
        self.h1.remove()
        self.h2.remove()


def list_conv_layers(module: nn.Module):
    layers = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            layers.append((name, m))
    return layers


def parse_label_file(label_file: Path):
    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.rsplit(maxsplit=2)
            if len(parts) != 3:
                continue

            img_path = Path(parts[0])
            big_label = int(parts[1])
            y = 0 if big_label == 2 else big_label
            samples.append((img_path, y))
    return samples


def build_transform(img_size, mean, std):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def find_last_conv_layer(module: nn.Module):
    last_conv_name = None
    last_conv_module = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv_name = name
            last_conv_module = m

    if last_conv_module is None:
        raise ValueError("模型中未找到 Conv2d 层，无法执行 Grad-CAM。")
    return last_conv_name, last_conv_module


def resolve_target_layer(module: nn.Module, layer_name: str | None):
    if layer_name is None:
        return find_last_conv_layer(module)

    target = dict(module.named_modules()).get(layer_name)
    if target is None:
        raise ValueError(f"未找到指定层: {layer_name}")
    if not isinstance(target, nn.Conv2d):
        raise ValueError(f"指定层 {layer_name} 不是 Conv2d，当前类型为 {type(target)}")
    return layer_name, target


def denormalize(img_tensor: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean, device=img_tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std_t + mean_t
    return img.clamp(0, 1)


def overlay_cam_on_image(image_01: np.ndarray, cam_01: np.ndarray, alpha=0.45):
    """
    image_01: HxWx3, [0,1]
    cam_01: HxW, [0,1]
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    heatmap = cmap(cam_01)[..., :3]  # HxWx3
    overlay = (1 - alpha) * image_01 + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def calc_border_focus_ratio(cam_01: np.ndarray, cam_threshold: float = 0.6, border_ratio: float = 0.15):
    """
    计算 CAM 高响应是否偏向边缘区域：
    - cam_threshold: 超过该阈值视为高响应像素
    - border_ratio: 每条边占图像宽/高的比例
    返回值越高，表示高响应更偏向边缘非中心区域
    """
    h, w = cam_01.shape
    bh = max(1, int(h * border_ratio))
    bw = max(1, int(w * border_ratio))

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    high_mask = cam_01 >= cam_threshold
    high_total = int(high_mask.sum())
    if high_total == 0:
        return 0.0
    high_in_border = int((high_mask & border_mask).sum())
    return float(high_in_border / high_total)


def save_vis_triplet(original, heatmap, overlay, out_path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)




def save_intermediate_results(pil_img: Image.Image, out_dir: Path):
    adv = data_cfg.get("advanced", {})
    t = T.Compose([T.Resize((data_cfg["img_size"], data_cfg["img_size"])), T.ToTensor()])
    x = t(pil_img)

    center_t = CenterWeightTransform(
        mode=adv.get("center_weight_mode", "none"),
        strength=float(adv.get("center_weight_strength", 0.0)),
        sigma=float(adv.get("center_weight_sigma", 0.45)),
    )
    border_t = BorderSuppressionAugment(
        prob=1.0,
        width_ratio=float(adv.get("border_aug_width_ratio", 0.12)),
        mode=adv.get("border_aug_mode", "blur"),
    )
    stain_t = MacenkoNormalizer(enabled=bool(adv.get("use_stain_normalization", False)))
    he_t = HEDeconvolution(enabled=True, output_mode="analysis")

    center_img = center_t(x.clone()).clamp(0, 1)
    border_img = border_t(x.clone()).clamp(0, 1)
    stain_img = stain_t(x.clone()).clamp(0, 1)
    _, he_info = he_t(x.clone())

    out_dir.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(center_img[:3]).save(out_dir / "center_weight.png")
    TF.to_pil_image(border_img[:3]).save(out_dir / "border_suppression.png")
    TF.to_pil_image(stain_img[:3]).save(out_dir / "stain_normalized.png")

    if he_info is not None:
        h = (he_info["H"] - he_info["H"].min()) / (he_info["H"].max() - he_info["H"].min() + 1e-8)
        e = (he_info["E"] - he_info["E"].min()) / (he_info["E"].max() - he_info["E"].min() + 1e-8)
        TF.to_pil_image(h).save(out_dir / "he_H_channel.png")
        TF.to_pil_image(e).save(out_dir / "he_E_channel.png")

def run_gradcam_for_one_label_file(
    model: nn.Module,
    checkpoint_path: Path,
    label_file: Path,
    out_dir: Path,
    target_name: str,
    target_layer: nn.Module,
    class_idx: int | None,
    max_samples: int,
    cam_threshold: float,
    border_ratio: float,
):
    gradcam = GradCAM(model, target_layer)
    samples = parse_label_file(label_file)
    if max_samples > 0:
        samples = samples[:max_samples]

    transform = build_transform(data_cfg["img_size"], data_cfg["mean"], data_cfg["std"])
    records = []
    heat_dir = out_dir / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, true_label) in enumerate(tqdm(samples, desc=f"Grad-CAM ({label_file.stem})", ncols=100)):
        if not img_path.exists():
            print(f"[WARN] 图片不存在，跳过: {img_path}")
            continue

        pil = Image.open(img_path).convert("RGB")
        if i == 0:
            save_intermediate_results(pil, out_dir / "intermediates")
        x = transform(pil).unsqueeze(0).to(next(model.parameters()).device)
        cam, logits = gradcam(x, class_idx=class_idx)

        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(prob))
        explained_class = class_idx if class_idx is not None else pred

        cam_up = torch.nn.functional.interpolate(
            cam,
            size=(data_cfg["img_size"], data_cfg["img_size"]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        img_denorm = denormalize(x.detach().cpu(), data_cfg["mean"], data_cfg["std"])[0]
        img_np = img_denorm.permute(1, 2, 0).numpy()
        cam_np = cam_up.detach().cpu().numpy()
        heatmap, overlay = overlay_cam_on_image(img_np, cam_np)

        border_focus_ratio_val = calc_border_focus_ratio(
            cam_np,
            cam_threshold=cam_threshold,
            border_ratio=border_ratio,
        )

        stem = img_path.stem
        out_img = heat_dir / f"{stem}_cam.png"
        save_vis_triplet(img_np, heatmap, overlay, out_img)

        record = {
            "image_path": str(img_path),
            "true_label": int(true_label),
            "pred_label": pred,
            "explained_class": int(explained_class),
            "correct": int(pred == true_label),
            "border_focus_ratio": float(border_focus_ratio_val),
            "output_image": str(out_img),
        }
        for c in range(len(prob)):
            record[f"prob_class_{c}"] = float(prob[c])
        records.append(record)

    gradcam.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gradcam_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "label_file": str(label_file),
        "num_samples": len(records),
        "target_layer": target_name,
        "class_idx_mode": "predicted" if class_idx is None else int(class_idx),
        "accuracy": float(np.mean([r["correct"] for r in records])) if records else None,
        "mean_border_focus_ratio": float(np.mean([r["border_focus_ratio"] for r in records])) if records else None,
        "cam_threshold": float(cam_threshold),
        "border_ratio": float(border_ratio),
        "output_dir": str(out_dir),
        "results_csv": str(csv_path),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, csv_path


def main():
    parser = argparse.ArgumentParser(description="在指定数据集上生成分类模型 Grad-CAM 热力图")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint 路径")
    parser.add_argument(
        "--label-file",
        type=str,
        nargs="+",
        required=True,
        help="标签 txt 文件路径（可一次输入多个）",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="输出根目录")
    parser.add_argument("--target-layer", type=str, default=None, help="目标卷积层名，例如 model.conv_head")
    parser.add_argument("--list-conv-layers", action="store_true", help="仅打印所有可用 Conv2d 层名并退出")
    parser.add_argument("--class-idx", type=int, default=None, help="固定解释某个类别；默认解释模型预测类别")
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理样本数，0 表示全部")
    parser.add_argument("--cam-threshold", type=float, default=0.6, help="统计边缘关注度时，高响应阈值")
    parser.add_argument("--border-ratio", type=float, default=0.15, help="统计边缘关注度时，边缘宽度占比")
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="可选：用于对比的 baseline checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    label_files = [Path(x) for x in args.label_file]
    out_root_dir = Path(args.out_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    for label_file in label_files:
        if not label_file.exists():
            raise FileNotFoundError(f"找不到 label file: {label_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    num_classes = data_cfg["num_classes"]
    core = LitSingleCell(model_cfg, num_classes=num_classes, data_advanced_cfg=data_cfg.get("advanced", {}))
    wrapper = _Wrapper.load_from_checkpoint(str(ckpt_path), core=core, map_location="cpu")
    model = wrapper.core.to(device)
    model.eval()

    conv_layers = list_conv_layers(model)
    if args.list_conv_layers:
        print("[Grad-CAM] 可用 Conv2d 层：")
        for i, (name, layer) in enumerate(conv_layers):
            print(f"  [{i:02d}] {name}: {layer}")
        return

    # 2) 选取目标层
    target_name, target_layer = resolve_target_layer(model, args.target_layer)
    print(f"[Grad-CAM] target layer: {target_name}")

    all_summaries = []
    for label_file in label_files:
        ds_name = label_file.stem
        ds_out_dir = out_root_dir / ds_name
        print(f"\n[Grad-CAM] 开始处理数据集: {label_file}")
        print(f"[Grad-CAM] 输出目录: {ds_out_dir}")

        summary, csv_path = run_gradcam_for_one_label_file(
            model=model,
            checkpoint_path=ckpt_path,
            label_file=label_file,
            out_dir=ds_out_dir,
            target_name=target_name,
            target_layer=target_layer,
            class_idx=args.class_idx,
            max_samples=args.max_samples,
            cam_threshold=args.cam_threshold,
            border_ratio=args.border_ratio,
        )
        all_summaries.append(summary)

        print("✅ 子数据集 Grad-CAM 完成")
        print(f"   样本数: {summary['num_samples']}")
        print(f"   结果 CSV: {csv_path}")
        print(f"   汇总 JSON: {ds_out_dir / 'summary.json'}")

    with open(out_root_dir / "summary_all_datasets.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 全部完成，总结文件: {out_root_dir / 'summary_all_datasets.json'}")



    if args.baseline_ckpt:
        baseline_core = LitSingleCell(model_cfg, num_classes=data_cfg["num_classes"], data_advanced_cfg=data_cfg.get("advanced", {}))
        baseline_wrapper = _Wrapper(baseline_core)
        baseline_sd = torch.load(args.baseline_ckpt, map_location="cpu")
        baseline_wrapper.load_state_dict(baseline_sd["state_dict"] if "state_dict" in baseline_sd else baseline_sd, strict=False)
        baseline_model = baseline_wrapper.core.eval().to(device)

        bsum, _ = run_gradcam_for_one_label_file(
            model=baseline_model,
            checkpoint_path=Path(args.baseline_ckpt),
            label_file=label_files[0],
            out_dir=out_root_dir / "baseline_compare",
            target_name=target_name,
            target_layer=target_layer,
            class_idx=args.class_idx,
            max_samples=args.max_samples,
            cam_threshold=args.cam_threshold,
            border_ratio=args.border_ratio,
        )
        compare = {
            "new_mean_border_focus_ratio": all_summaries[0].get("mean_border_focus_ratio") if all_summaries else None,
            "baseline_mean_border_focus_ratio": bsum.get("mean_border_focus_ratio"),
            "delta": None,
        }
        if compare["new_mean_border_focus_ratio"] is not None and compare["baseline_mean_border_focus_ratio"] is not None:
            compare["delta"] = compare["new_mean_border_focus_ratio"] - compare["baseline_mean_border_focus_ratio"]
        with open(out_root_dir / "compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(compare, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

'''
python tools/gradcam.py \
  --ckpt "/root/autodl-tmp/projects/myq/SingleCellProject/outputs/260323_gt2yolo_576_0.65_2class_onlineAug/epoch=23-val_acc_macro=0.0000.ckpt" \
  --label-file "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU_labels_16.txt" \
  --out-dir "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_gradcam/debug_layers" \
  --list-conv-layers
'''

'''
python tools/gradcam.py \
  --ckpt "/root/autodl-tmp/projects/myq/SingleCellProject/outputs/260323_gt2yolo_576_0.65_2class_onlineAug/epoch=23-val_acc_macro=0.0000.ckpt" \
  --label-file \
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU_labels_16.txt" \
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH_labels_16.txt" \
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL_labels_16.txt" \
  --out-dir "/root/autodl-tmp/projects/myq/SingleCellProject/outputs_gradcam/batch_debug" \
  --target-layer "model.conv_head" \
  --cam-threshold 0.6 \
  --border-ratio 0.15 \
  --max-samples 100
'''