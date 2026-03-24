#无可解释性 有/无预训练
#13版 改用了torchvison
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T


class LabelFileDataset(Dataset):
    def __init__(self, label_file, img_size, mean, std, augment=None, training=True):
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.rsplit(maxsplit=2)
                if len(parts) != 3:
                    print(f"[WARN] bad line: {s}")
                    continue

                img = parts[0]              # 图片路径
                big_label = int(parts[1])   # 第二列，大类
                small_label = int(parts[2]) # 第三列，小类（现在不用）

                # 保持你原来的逻辑：大类 2 → 0，其它不变
                y = 0 if big_label == 2 else big_label
                # y = small_label
                self.samples.append((img, y))

        self.img_size = img_size
        self.training = training
        self.transform = self._build_transform(img_size, mean, std, augment, training)

    def _build_transform(self, img_size, mean, std, augment, training):
        t = []

        if training and augment:
            # 如果配置了 random_crop_scale，就用 RandomResizedCrop，
            # 否则在后面统一 Resize
            random_crop_scale = augment.get("random_crop_scale", None)
            if random_crop_scale is not None:
                t.append(
                    T.RandomResizedCrop(
                        size=img_size,
                        scale=tuple(random_crop_scale),
                        ratio=(0.9, 1.1),
                    )
                )
            else:
                t.append(T.Resize((img_size, img_size)))

            # 水平翻转
            hflip_p = augment.get("hflip_p", 0.0)
            if hflip_p > 0:
                t.append(T.RandomHorizontalFlip(p=hflip_p))

            # 垂直翻转
            vflip_p = augment.get("vflip_p", 0.0)
            if vflip_p > 0:
                t.append(T.RandomVerticalFlip(p=vflip_p))

            # 旋转
            rotate_deg = augment.get("rotate_deg", 0)
            if rotate_deg > 0:
                t.append(T.RandomRotation(degrees=rotate_deg))

            # 颜色抖动
            if any(k in augment for k in ["brightness", "contrast", "saturation", "hue"]):
                t.append(
                    T.ColorJitter(
                        brightness=augment.get("brightness", 0),
                        contrast=augment.get("contrast", 0),
                        saturation=augment.get("saturation", 0),
                        hue=augment.get("hue", 0),
                    )
                )
        else:
            # 验证 / 不使用增广：只做 Resize
            t.append(T.Resize((img_size, img_size)))

        # 转 tensor & Normalize
        # ToTensor(): HWC[0-255] -> CHW[0-1]
        # 然后 Normalize(mean, std)（mean/std 也是按 0~1 设）
        t.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        return T.Compose(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, y
        except Exception as e:
            print(f"\n[ERROR] Failed to load: {path} | Error: {e}")
            # 兜底：坏图时返回全零图，避免训练直接挂掉
            dummy = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            return dummy, y


'''
#12.30可解释性v2 src/datasets.py
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import csv
import os
import numpy as np

class LabelFileDataset(Dataset):
    def __init__(self, label_file, feature_csv_path, img_size, mean, std, augment=None, training=True):
        """
        label_file: 增强后的标签文件
        feature_csv_path: 特征 CSV 文件路径
        """
        self.samples = []
        
        # 1. 读取 Label txt
        print(f"[Dataset] Loading labels from {label_file}")
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                parts = s.rsplit(maxsplit=2)
                if len(parts) != 3: continue

                img_path = parts[0]
                big_label = int(parts[1])
                y = 0 if big_label == 2 else big_label
                self.samples.append((img_path, y))

        # 2. 读取 Feature CSV
        self.feature_dict = {}
        if feature_csv_path and os.path.exists(feature_csv_path):
            print(f"[Dataset] Loading features from {feature_csv_path} ...")
            with open(feature_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row['filename']
                    key = os.path.splitext(fname)[0] # 去掉 .png
                    
                    # 提取数值特征 (5维)
                    # 在这里只做提取，不做归一化，方便后面加噪声
                    feats = [
                        float(row['area']),
                        float(row['perimeter']),
                        float(row['circularity']),
                        float(row['nucleus_area']),
                        float(row['nc_ratio'])
                    ]
                    self.feature_dict[key] = np.array(feats, dtype=np.float32)
            print(f"[Dataset] Loaded {len(self.feature_dict)} feature entries.")
        else:
            print(f"⚠️ [Dataset] Warning: Feature CSV not found: {feature_csv_path}")

        self.img_size = img_size
        self.training = training
        self.transform = self._build_transform(img_size, mean, std, augment, training)

    def _build_transform(self, img_size, mean, std, augment, training):
        t = []
        if training and augment:
            t.append(T.Resize((img_size, img_size)))
            if augment.get("hflip_p", 0) > 0:
                t.append(T.RandomHorizontalFlip(p=augment.get("hflip_p")))
            if augment.get("vflip_p", 0) > 0:
                t.append(T.RandomVerticalFlip(p=augment.get("vflip_p")))
            if any(k in augment for k in ["brightness", "contrast", "saturation", "hue"]):
                t.append(T.ColorJitter(
                    brightness=augment.get("brightness", 0),
                    contrast=augment.get("contrast", 0),
                    saturation=augment.get("saturation", 0),
                    hue=augment.get("hue", 0)
                ))
        else:
            t.append(T.Resize((img_size, img_size)))

        t.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        return T.Compose(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        
        # --- 图片处理 ---
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"[Err] {path}: {e}")
            img = torch.zeros(3, self.img_size, self.img_size)

        # --- 特征提取 ---
        filename_with_aug = os.path.splitext(os.path.basename(path))[0] 
        original_key = filename_with_aug.rsplit('_', 1)[0]
        
        # 获取原始数值
        raw_feats = self.feature_dict.get(original_key, np.zeros(5, dtype=np.float32))
        
        # === 关键修改：特征增强 (Feature Jittering) ===
        if self.training:
            # 生成与特征同维度的随机噪声 (均值0，标准差为特征值的 5%)
            # 这样既能增加波动，又不会改变特征的数量级
            noise = np.random.normal(0, 0.05 * np.abs(raw_feats), size=raw_feats.shape)
            feats_aug = raw_feats + noise
        else:
            feats_aug = raw_feats

        # === 再次做 Log 处理 ===
        # 面积类数值大，取Log；比例类数值小，保持原样
        # [area, peri, circ, n_area, nc_ratio]
        final_feats = [
            np.log1p(max(0, feats_aug[0])), # Log Area (防止负数)
            np.log1p(max(0, feats_aug[1])), # Log Perimeter
            feats_aug[2],                   # Circularity
            np.log1p(max(0, feats_aug[3])), # Log Nucleus Area
            feats_aug[4]                    # NC Ratio
        ]

        features_tensor = torch.tensor(final_feats, dtype=torch.float32)

        return img, features_tensor, y
'''