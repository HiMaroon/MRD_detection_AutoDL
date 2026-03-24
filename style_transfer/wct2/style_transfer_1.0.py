"""
single_cell_style_transfer_run.py
直接修改配置运行的单细胞风格迁移脚本
"""

import os
from typing import Optional
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import pickle
import math

from transfer import WCT2


# ==================== 配置区域（直接修改这里） ====================

CONFIG = {
    # 输入路径
    'center_a_dir': '/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/train',      # 中心A单细胞图像文件夹（风格源）
    'center_b_dir': '/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_TJMU',      # 中心B单细胞图像文件夹（待迁移）
    'output_dir': '/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/test_TJMU_transfer',       # 输出文件夹（迁移后的图像）
    
    # 模型参数
    'model_path': './model_checkpoints',           # WCT2模型路径
    'device': 'cuda:0',                            # 设备：'cuda:0' 或 'cpu'
    
    # 图像尺寸
    'target_size': 576,                            # 目标尺寸（224, 576等，会自动调整为16的倍数）
    
    # 迁移参数
    'alpha': 0.3,                                  # 风格强度（0-1）：0.1-0.3保守，0.4-0.6平衡，0.7-1.0激进
    'use_histogram': True,                         # 是否使用直方图预处理
    
    # 风格库参数
    'max_style_samples': 25608,                      # 风格库最大样本数（越多越慢但更稳定）
    'cache_dir': './cache',                        # 缓存目录
    
    # 其他
    'top_k_match': 5,                              # 风格匹配时Top-K随机选择
}


# ==================== 核心代码（无需修改） ====================

def make_divisible_by_16(size: int) -> int:
    """确保尺寸能被16整除（VGG下采样要求）"""
    return math.ceil(size / 16) * 16


class SingleCellStyleTransfer:
    """
    单细胞图像风格迁移器
    """
    
    def __init__(self,
                 model_path: str = './model_checkpoints',
                 device: str = 'cuda:0',
                 cache_dir: str = './cache',
                 default_size: int = 224):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_size = default_size
        
        print(f"初始化WCT2模型 (device: {self.device}, default_size: {default_size})...")
        self.wct2 = WCT2(
            model_path=model_path,
            transfer_at=['decoder'],
            option_unpool='sum',
            device=self.device,
            verbose=False
        )
        
        # 按尺寸分类存储风格库
        self.style_banks = {}
        
    def _compute_histogram(self, img: np.ndarray, bins: int = 16) -> np.ndarray:
        """计算颜色直方图"""
        hist = []
        for i in range(3):
            h = cv2.calcHist([img], [i], None, [bins], [0, 256])
            h = h / (h.sum() + 1e-7)
            hist.extend(h.flatten())
        return np.array(hist, dtype=np.float32)
    
    def histogram_match(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        """快速直方图匹配"""
        content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(np.float32)
        style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        matched = content_lab.copy()
        for i in range(3):
            content_flat = content_lab[:, :, i].ravel()
            style_flat = style_lab[:, :, i].ravel()
            
            content_sorted = np.sort(content_flat)
            style_sorted = np.sort(style_flat)
            
            indices = np.searchsorted(content_sorted, content_flat)
            indices = np.clip(indices, 0, len(style_sorted) - 1)
            matched_flat = style_sorted[indices]
            matched[:, :, i] = matched_flat.reshape(content_lab[:, :, i].shape)
        
        return cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def wct2_transfer(self, content: np.ndarray, style: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """WCT2神经风格迁移"""
        def to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return img.to(self.device)
        
        content_t = to_tensor(content)
        style_t = to_tensor(style)
        
        with torch.no_grad():
            result = self.wct2.transfer(
                content_t, style_t,
                np.asarray([]), np.asarray([]),
                alpha=alpha
            )
        
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
    
    def build_style_bank(self, 
                         center_a_dir: str, 
                         target_size: Optional[int] = None,
                         max_samples: int = 500):
        """构建指定尺寸的风格库"""
        size = target_size or self.default_size
        size = make_divisible_by_16(size)
        
        print(f"\n构建风格库: {center_a_dir} @ {size}x{size}")
        
        image_paths = list(Path(center_a_dir).glob('*.jpg')) + \
                     list(Path(center_a_dir).glob('*.png')) + \
                     list(Path(center_a_dir).glob('*.jpeg'))
        
        if len(image_paths) == 0:
            raise ValueError(f"未找到图像: {center_a_dir}")
        
        print(f"发现 {len(image_paths)} 张图像")
        
        if len(image_paths) > max_samples:
            image_paths = random.sample(image_paths, max_samples)
            print(f"随机采样至 {max_samples} 张")
        
        style_images = []
        style_histograms = []
        
        for path in tqdm(image_paths, desc=f"构建风格库"):
            img = cv2.imread(str(path))
            if img is None:
                continue
            
            img = cv2.resize(img, (size, size))
            hist = self._compute_histogram(img)
            
            style_images.append(img)
            style_histograms.append(hist)
        
        self.style_banks[size] = {
            'images': style_images,
            'histograms': style_histograms
        }
        
        # 保存缓存
        cache_file = self.cache_dir / f'style_bank_{size}.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(self.style_banks[size], f)
        
        print(f"风格库构建完成: {len(style_images)} 个样本 @ {size}x{size}")
        return size
    
    def load_style_bank(self, size: int):
        """从缓存加载风格库"""
        cache_file = self.cache_dir / f'style_bank_{size}.pkl'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.style_banks[size] = pickle.load(f)
            print(f"风格库加载完成: {len(self.style_banks[size]['images'])} 个样本 @ {size}x{size}")
            return True
        return False
    
    def find_best_style(self, 
                        content_img: np.ndarray, 
                        target_size: int,
                        top_k: int = 5) -> np.ndarray:
        """查找最佳风格"""
        size = make_divisible_by_16(target_size)
        
        if size not in self.style_banks:
            cache_file = self.cache_dir / f'style_bank_{size}.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.style_banks[size] = pickle.load(f)
            else:
                raise ValueError(f"没有找到 {size}x{size} 的风格库，请先构建")
        
        bank = self.style_banks[size]
        content_hist = self._compute_histogram(content_img)
        
        similarities = []
        for style_hist in bank['histograms']:
            sim = cv2.compareHist(content_hist, style_hist, cv2.HISTCMP_CORREL)
            similarities.append(sim)
        
        top_indices = np.argsort(similarities)[-top_k:]
        best_idx = random.choice(top_indices)
        
        return bank['images'][best_idx]
    
    def transfer_single(self,
                       content_path: str,
                       output_path: str,
                       target_size: int,
                       alpha: float = 0.7,
                       use_histogram: bool = True,
                       top_k: int = 5) -> str:
        """迁移单张图像"""
        content = cv2.imread(content_path)
        if content is None:
            raise ValueError(f"无法读取: {content_path}")
        
        size = make_divisible_by_16(target_size)
        
        # Resize到目标尺寸
        content_resized = cv2.resize(content, (size, size))
        
        # 查找风格
        style = self.find_best_style(content_resized, size, top_k=top_k)
        
        # 直方图匹配
        if use_histogram:
            content_resized = self.histogram_match(content_resized, style)
        
        # WCT2迁移
        result = self.wct2_transfer(content_resized, style, alpha=alpha)
        
        # 保存
        cv2.imwrite(output_path, result)
        
        return output_path
    
    def transfer_batch(self,
                       center_b_dir: str,
                       output_dir: str,
                       target_size: int,
                       alpha: float = 0.7,
                       use_histogram: bool = True,
                       top_k: int = 5):
        """批量迁移"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = list(Path(center_b_dir).glob('*.jpg')) + \
                     list(Path(center_b_dir).glob('*.png')) + \
                     list(Path(center_b_dir).glob('*.jpeg'))
        
        if len(image_paths) == 0:
            raise ValueError(f"未找到图像: {center_b_dir}")
        
        size = make_divisible_by_16(target_size)
        print(f"\n批量迁移设置:")
        print(f"  输入: {center_b_dir}")
        print(f"  输出: {output_dir}")
        print(f"  图像数: {len(image_paths)}")
        print(f"  尺寸: {size}x{size}")
        print(f"  alpha: {alpha}")
        print(f"  直方图预处理: {use_histogram}")
        
        # 检查风格库
        if size not in self.style_banks:
            if not self.load_style_bank(size):
                raise ValueError(f"请先构建 {size}x{size} 的风格库")
        
        # 批量处理
        success = 0
        failed = []
        
        for img_path in tqdm(image_paths, desc="风格迁移"):
            output_path = os.path.join(output_dir, img_path.name)
            try:
                self.transfer_single(
                    str(img_path), output_path,
                    target_size=size,
                    alpha=alpha,
                    use_histogram=use_histogram,
                    top_k=top_k
                )
                success += 1
            except Exception as e:
                failed.append((str(img_path), str(e)))
        
        print(f"\n{'='*50}")
        print(f"迁移完成: {success}/{len(image_paths)} 张成功")
        if failed:
            print(f"失败 {len(failed)} 张:")
            for path, err in failed[:5]:
                print(f"  {path}: {err}")
        print(f"{'='*50}")
        
        return output_dir


def main():
    """主函数：读取配置并执行"""
    cfg = CONFIG
    
    print("="*60)
    print("单细胞图像风格迁移")
    print("中心B → 中心A 风格")
    print("="*60)
    
    # 检查路径
    if not os.path.exists(cfg['center_a_dir']):
        raise ValueError(f"中心A文件夹不存在: {cfg['center_a_dir']}")
    if not os.path.exists(cfg['center_b_dir']):
        raise ValueError(f"中心B文件夹不存在: {cfg['center_b_dir']}")
    
    # 调整尺寸
    size = make_divisible_by_16(cfg['target_size'])
    if size != cfg['target_size']:
        print(f"尺寸调整: {cfg['target_size']} → {size} (16的倍数)")
    
    # 初始化
    transfer = SingleCellStyleTransfer(
        model_path=cfg['model_path'],
        device=cfg['device'],
        cache_dir=cfg['cache_dir'],
        default_size=size
    )
    
    # 构建或加载风格库
    cache_file = Path(cfg['cache_dir']) / f'style_bank_{size}.pkl'
    if cache_file.exists():
        print(f"\n发现缓存的风格库，直接加载...")
        transfer.load_style_bank(size)
    else:
        print(f"\n构建新的风格库...")
        transfer.build_style_bank(
            cfg['center_a_dir'], 
            target_size=size,
            max_samples=cfg['max_style_samples']
        )
    
    # 执行迁移
    transfer.transfer_batch(
        center_b_dir=cfg['center_b_dir'],
        output_dir=cfg['output_dir'],
        target_size=size,
        alpha=cfg['alpha'],
        use_histogram=cfg['use_histogram'],
        top_k=cfg['top_k_match']
    )
    
    print(f"\n输出位置: {cfg['output_dir']}")
    print("完成！")


if __name__ == '__main__':
    main()