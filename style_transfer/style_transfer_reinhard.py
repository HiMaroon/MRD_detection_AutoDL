# -*- coding: utf-8 -*-
import os
import json
import random
import multiprocessing as mp
from glob import glob
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage import color, exposure

# =========================================================
# 1. 配置
# =========================================================
TRAIN_DIRS = [
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val",
]

TEST_DIRS = {
    "test_BJH": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH",
    "test_FXH_noALL": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL",
    "test_TJMU": "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU",
}

OUT_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer_reinhard"
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

SEED = 42
RECURSIVE = True
N_REF_IMAGES = 30000          #参考图像数量
SAVE_REINHARD = True
SAVE_REINHARD_HIST = True     # 是否执行第二步直方图匹配

# ⚡️ 性能关键配置
NUM_WORKERS = max(1, mp.cpu_count() - 2)  # 保留2个核心给系统，其余全用上
CHUNK_SIZE = 50             # 每个进程一次处理多少张图片，减少进程间通信开销

# =========================================================
# 2. 通用函数
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder, recursive=True):
    files = []
    if recursive:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, "**", ext), recursive=True))
    else:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, ext)))
    return sorted(set(files))

def collect_images(folders):
    all_files = []
    for folder in folders:
        if os.path.isdir(folder):
            all_files.extend(list_images(folder, recursive=RECURSIVE))
        else:
            print(f"⚠️ 无效目录，跳过: {folder}")
    return sorted(set(all_files))

def read_rgb_fast(path):
    """快速读取并转换为 RGB numpy 数组"""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def save_rgb_fast(arr, path):
    """快速保存"""
    try:
        # 确保数据类型正确
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    except Exception as e:
        print(f"Error saving {path}: {e}")

def relative_name_from_root(src_path, root_hint):
    rel = os.path.relpath(src_path, root_hint)
    return rel.replace("\\", "/")

def safe_out_path(base_dir, rel_path):
    out_path = os.path.join(base_dir, rel_path)
    ensure_dir(os.path.dirname(out_path))
    return out_path

# =========================================================
# 3. 核心算法 (保持纯函数，便于多进程调用)
# =========================================================
def apply_reinhard_single(rgb_uint8, target_mean, target_std):
    """
    对单张图片应用 Reinhard 标准化
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    
    # 计算当前图像的统计量
    # reshape(-1, 3) 展平像素，axis=0 计算通道均值
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = lab.reshape(-1, 3).std(axis=0) + 1e-6
    
    # 标准化并映射到目标分布
    out_lab = (lab - src_mean) / src_std
    out_lab = out_lab * target_std + target_mean
    
    out_rgb = color.lab2rgb(out_lab)
    return np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)

def apply_hist_match_single(src_uint8, ref_uint8):
    """
    直方图匹配
    """
    matched = exposure.match_histograms(src_uint8, ref_uint8, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)

# =========================================================
# 4. 多进程工作函数
# =========================================================
def process_batch(args):
    """
    供 multiprocessing.Pool 调用的批次处理函数
    args: (batch_paths, test_dir, out_reinhard_dir, out_hist_dir, 
           tgt_mean, tgt_std, style_pool, save_r, save_rh)
    """
    batch_paths, test_dir, out_r_dir, out_h_dir, tgt_mean, tgt_std, style_pool, save_r, save_rh = args
    
    results = []
    
    # 预加载所有可能的参考图像到内存？不，style_pool可能太大。
    # 优化策略：在每个批次内，如果 style_pool 不大，可以缓存；如果很大，每次随机读。
    # 这里假设 style_pool 已经加载到内存中作为路径列表，我们需要读取它们。
    # 为了极致速度，建议 style_pool 不要太大，或者预先读取少量代表图。
    # 此处保持原逻辑：随机选一张参考图。
    
    for img_path in batch_paths:
        rgb = read_rgb_fast(img_path)
        if rgb is None:
            continue
            
        rel_name = relative_name_from_root(img_path, test_dir)
        
        # 随机选择参考图
        ref_path = random.choice(style_pool)
        ref_rgb = read_rgb_fast(ref_path)
        if ref_rgb is None:
            continue

        # 1. Reinhard
        reinhard_img = apply_reinhard_single(rgb, tgt_mean, tgt_std)
        r_out_path = ""
        
        if save_r:
            r_out_path = safe_out_path(out_r_dir, rel_name)
            save_rgb_fast(reinhard_img, r_out_path)

        # 2. Hist Match
        h_out_path = ""
        if save_rh:
            hist_img = apply_hist_match_single(reinhard_img, ref_rgb)
            h_out_path = safe_out_path(out_h_dir, rel_name)
            save_rgb_fast(hist_img, h_out_path)
            
        results.append({
            "original_path": img_path,
            "relative_name": rel_name,
            "reference_image": ref_path,
            "reinhard_out": r_out_path,
            "reinhard_hist_out": h_out_path,
        })
        
    return results

# =========================================================
# 5. 参考库构建 (单次运行，无需并行)
# =========================================================
def build_reference_pool():
    print("📂 Collecting training images for reference...")
    style_all = collect_images(TRAIN_DIRS)
    if len(style_all) == 0:
        raise ValueError("未找到 train/val 风格参考图像")

    n = min(N_REF_IMAGES, len(style_all))
    print(f"🎲 Randomly selecting {n} reference images...")
    selected = random.sample(style_all, n)

    ref_dir = os.path.join(OUT_ROOT, "reference")
    ensure_dir(ref_dir)

    # 保存文件列表
    with open(os.path.join(ref_dir, "selected_style_images.txt"), "w", encoding="utf-8") as f:
        for p in selected:
            f.write(p + "\n")

    print("📊 Computing Reinhard template statistics...")
    lab_means = []
    lab_stds = []
    
    # 这里也可以并行，但 N_REF_IMAGES=30k 较快，暂不复杂化
    for p in tqdm(selected, desc="Computing Stats"):
        rgb = read_rgb_fast(p)
        if rgb is None: continue
        rgb_f = rgb.astype(np.float32) / 255.0
        lab = color.rgb2lab(rgb_f)
        mean = lab.reshape(-1, 3).mean(axis=0)
        std = lab.reshape(-1, 3).std(axis=0) + 1e-6
        lab_means.append(mean)
        lab_stds.append(std)

    if not lab_means:
        raise ValueError("No valid images found for template computation")

    template = {
        "mean": np.mean(lab_means, axis=0).tolist(),
        "std": np.mean(lab_stds, axis=0).tolist(),
    }
    
    with open(os.path.join(ref_dir, "ref_mean_std_lab.json"), "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    return selected, template

# =========================================================
# 6. 主处理流程 (并行化)
# =========================================================
def process_one_testset_parallel(test_name, test_dir, style_pool, template):
    print(f"\n🚀 Processing {test_name}: {test_dir} (Workers: {NUM_WORKERS})")

    test_imgs = list_images(test_dir, recursive=RECURSIVE)
    if len(test_imgs) == 0:
        print(f"⚠️ {test_name} 没有图像，跳过")
        return

    out_base = os.path.join(OUT_ROOT, test_name)
    out_reinhard = os.path.join(out_base, "reinhard")
    out_reinhard_hist = os.path.join(out_base, "reinhard_hist")
    ensure_dir(out_reinhard)
    ensure_dir(out_reinhard_hist)

    tgt_mean = np.array(template["mean"], dtype=np.float32)
    tgt_std = np.array(template["std"], dtype=np.float32)

    # 准备批次数据
    # 将图片列表分割成 chunks
    chunks = [test_imgs[i:i + CHUNK_SIZE] for i in range(0, len(test_imgs), CHUNK_SIZE)]
    
    # 构造参数列表
    task_args = [
        (chunk, test_dir, out_reinhard, out_reinhard_hist, tgt_mean, tgt_std, style_pool, SAVE_REINHARD, SAVE_REINHARD_HIST)
        for chunk in chunks
    ]

    all_results = []
    
    # 使用进程池并行处理
    # 注意：style_pool 是一个列表，会被 pickle 序列化传递给子进程。
    # 如果 style_pool 很大（30k路径），序列化开销较大。
    # 优化：可以将 style_pool 写入临时文件，在子进程中读取，或者使用 initializer。
    # 鉴于路径字符串较轻，直接传递通常可接受。若极慢，可改用 multiprocessing.Manager 或全局变量初始化。
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # imap_unordered 比 map 更快，因为它只要有一个结果回来就 yield，配合 tqdm 显示进度
        for batch_results in tqdm(pool.imap_unordered(process_batch, task_args), total=len(chunks), desc=f"{test_name} Progress"):
            all_results.extend(batch_results)

    # 保存元数据
    if all_results:
        df = pd.DataFrame(all_results)
        df.insert(0, "test_name", test_name)
        csv_path = os.path.join(out_base, "metadata.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Metadata saved to {csv_path}")
    
    print(f"✅ Done: {test_name}")

# =========================================================
# 7. 主程序
# =========================================================
def main():
    set_seed(SEED)
    ensure_dir(OUT_ROOT)

    # 1. 构建参考库 (串行)
    style_pool, template = build_reference_pool()
    
    # 打印参考库信息
    print(f"Reference Pool Size: {len(style_pool)}")
    print(f"Target Mean (LAB): {template['mean']}")
    print(f"Target Std (LAB): {template['std']}")

    # 2. 并行处理每个测试集
    for test_name, test_dir in TEST_DIRS.items():
        process_one_testset_parallel(test_name, test_dir, style_pool, template)

    print("\n🎉 All tasks finished.")

if __name__ == "__main__":
    # Linux/macOS 需要保护入口点
    main()