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


def main():
    parser = argparse.ArgumentParser(description="在指定数据集上生成分类模型 Grad-CAM 热力图")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint 路径")
    parser.add_argument("--label-file", type=str, required=True, help="标签 txt 文件路径")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target-layer", type=str, default=None, help="目标卷积层名，例如 model.conv_head")
    parser.add_argument("--class-idx", type=int, default=None, help="固定解释某个类别；默认解释模型预测类别")
    parser.add_argument("--max-samples", type=int, default=0, help="最多处理样本数，0 表示全部")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    label_file = Path(args.label_file)
    out_dir = Path(args.out_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if not label_file.exists():
        raise FileNotFoundError(f"找不到 label file: {label_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    num_classes = data_cfg["num_classes"]
    core = LitSingleCell(model_cfg, num_classes=num_classes)
    wrapper = _Wrapper.load_from_checkpoint(str(ckpt_path), core=core, map_location="cpu")
    model = wrapper.core.to(device)
    model.eval()

    # 2) 选取目标层
    target_name, target_layer = resolve_target_layer(model, args.target_layer)
    print(f"[Grad-CAM] target layer: {target_name}")

    gradcam = GradCAM(model, target_layer)

    # 3) 数据准备
    samples = parse_label_file(label_file)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    transform = build_transform(data_cfg["img_size"], data_cfg["mean"], data_cfg["std"])

    records = []
    heat_dir = out_dir / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    # 4) 遍历样本生成 CAM
    for img_path, true_label in tqdm(samples, desc="Grad-CAM", ncols=100):
        if not img_path.exists():
            print(f"[WARN] 图片不存在，跳过: {img_path}")
            continue

        pil = Image.open(img_path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(device)

        cam, logits = gradcam(x, class_idx=args.class_idx)

        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(prob))
        explained_class = args.class_idx if args.class_idx is not None else pred

        # cam resize 回输入大小
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

        stem = img_path.stem
        out_img = heat_dir / f"{stem}_cam.png"
        save_vis_triplet(img_np, heatmap, overlay, out_img)

        record = {
            "image_path": str(img_path),
            "true_label": int(true_label),
            "pred_label": pred,
            "explained_class": int(explained_class),
            "correct": int(pred == true_label),
            "output_image": str(out_img),
        }
        for c in range(len(prob)):
            record[f"prob_class_{c}"] = float(prob[c])
        records.append(record)

    gradcam.close()

    # 5) 保存统计
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gradcam_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary = {
        "checkpoint": str(ckpt_path),
        "label_file": str(label_file),
        "num_samples": len(records),
        "target_layer": target_name,
        "class_idx_mode": "predicted" if args.class_idx is None else int(args.class_idx),
        "accuracy": float(np.mean([r["correct"] for r in records])) if records else None,
        "output_dir": str(out_dir),
        "results_csv": str(csv_path),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Grad-CAM 完成")
    print(f"   样本数: {summary['num_samples']}")
    print(f"   结果 CSV: {csv_path}")
    print(f"   汇总 JSON: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
