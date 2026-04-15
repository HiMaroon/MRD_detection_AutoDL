# -*- coding: utf-8 -*-
import os
import random
import hashlib
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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

OUT_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer_fda"
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

SEED = 42
RECURSIVE = True

# 参考池
N_REF_IMAGES = 30000     # 先从 train+val 中抽多少张做大参考池
REF_POOL_SIZE = 256      # 实际 worker 中用于匹配的小参考池，建议 128/256/512

# FDA 参数
BETAS = [0.01, 0.03, 0.05]

# 并行参数
NUM_WORKERS = min(8, os.cpu_count() or 1)
CHUNK_SIZE = 16

# =========================================================
# 2. 通用函数
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(folder: str, recursive: bool = True) -> List[str]:
    files = []
    if recursive:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, "**", ext), recursive=True))
    else:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, ext)))
    return sorted(set(files))

def collect_images(folders: List[str]) -> List[str]:
    all_files = []
    for folder in folders:
        if os.path.isdir(folder):
            all_files.extend(list_images(folder, recursive=RECURSIVE))
        else:
            print(f"⚠️ 无效目录，跳过: {folder}")
    return sorted(set(all_files))

def read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_rgb(arr: np.ndarray, path: str):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def relative_name_from_root(src_path: str, root_hint: str) -> str:
    rel = os.path.relpath(src_path, root_hint)
    return rel.replace("\\", "/")

def safe_out_path(base_dir: str, rel_path: str) -> str:
    out_path = os.path.join(base_dir, rel_path)
    ensure_dir(os.path.dirname(out_path))
    return out_path

def beta_to_name(beta: float) -> str:
    return f"beta_{str(beta).replace('.', 'p')}"

# =========================================================
# 3. FDA 核心
# =========================================================
def fft_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    img: H, W, C, uint8/float
    return: amplitude, phase
    """
    img = img.astype(np.float32)
    fft = np.fft.fft2(img, axes=(0, 1))
    amp = np.abs(fft)
    pha = np.angle(fft)
    return amp, pha

def low_freq_mutate(amp_src: np.ndarray, amp_trg: np.ndarray, beta: float) -> np.ndarray:
    """
    替换低频 amplitude
    beta 越大，替换区域越大
    """
    a_src = np.fft.fftshift(amp_src, axes=(0, 1)).copy()
    a_trg = np.fft.fftshift(amp_trg, axes=(0, 1))

    h, w, c = a_src.shape
    b = int(np.floor(min(h, w) * beta))

    if b <= 0:
        return amp_src.copy()

    ch, cw = h // 2, w // 2
    h1, h2 = max(ch - b, 0), min(ch + b + 1, h)
    w1, w2 = max(cw - b, 0), min(cw + b + 1, w)

    a_src[h1:h2, w1:w2, :] = a_trg[h1:h2, w1:w2, :]
    a_src = np.fft.ifftshift(a_src, axes=(0, 1))
    return a_src

def fda_from_precomputed(
    amp_src: np.ndarray,
    pha_src: np.ndarray,
    amp_trg: np.ndarray,
    beta: float = 0.03,
) -> np.ndarray:
    """
    使用预先计算好的 FFT 结果，避免重复 FFT
    """
    amp_src_mut = low_freq_mutate(amp_src, amp_trg, beta=beta)
    fft_mut = amp_src_mut * np.exp(1j * pha_src)
    out = np.fft.ifft2(fft_mut, axes=(0, 1))
    out = np.real(out)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# =========================================================
# 4. 参考池构建
# =========================================================
def build_reference_pool() -> List[str]:
    style_all = collect_images(TRAIN_DIRS)
    if len(style_all) == 0:
        raise ValueError("❌ 未找到 train/val 风格参考图像")

    n = min(N_REF_IMAGES, len(style_all))
    selected = random.sample(style_all, n)

    ref_dir = os.path.join(OUT_ROOT, "reference")
    ensure_dir(ref_dir)

    with open(os.path.join(ref_dir, "selected_style_images.txt"), "w", encoding="utf-8") as f:
        for p in selected:
            f.write(p + "\n")

    pd.DataFrame({"reference_image": selected}).to_csv(
        os.path.join(ref_dir, "reference_pool.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return selected

def choose_small_reference_pool(style_pool: List[str], pool_size: int) -> List[str]:
    n = min(pool_size, len(style_pool))
    return style_pool[:n]

# =========================================================
# 5. Worker 进程内全局缓存
# =========================================================
_WORKER_STYLE_POOL: List[str] = []
_WORKER_REF_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

def init_worker(style_pool_small: List[str]):
    """
    每个 worker 初始化时注入小参考池，并清空本进程缓存
    """
    global _WORKER_STYLE_POOL, _WORKER_REF_CACHE
    _WORKER_STYLE_POOL = style_pool_small
    _WORKER_REF_CACHE = {}

def stable_pick_reference(img_path: str) -> str:
    """
    对同一张测试图，稳定选取同一张参考图
    """
    if len(_WORKER_STYLE_POOL) == 0:
        raise ValueError("❌ worker 内参考池为空")
    h = hashlib.md5(img_path.encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(_WORKER_STYLE_POOL)
    return _WORKER_STYLE_POOL[idx]

def get_cached_ref_fft(ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    worker 内参考图 FFT 缓存
    """
    global _WORKER_REF_CACHE
    if ref_path not in _WORKER_REF_CACHE:
        ref_rgb = read_rgb(ref_path)
        _WORKER_REF_CACHE[ref_path] = fft_image(ref_rgb)
    return _WORKER_REF_CACHE[ref_path]

# =========================================================
# 6. 单张图处理（并行 worker）
# =========================================================
def process_one_image_worker(args):
    img_path, test_dir, out_base, betas = args

    rel_name = relative_name_from_root(img_path, test_dir)
    src_rgb = read_rgb(img_path)

    # source FFT 只算一次
    amp_src, pha_src = fft_image(src_rgb)

    # 稳定选参考图，并使用 worker 内缓存
    ref_path = stable_pick_reference(img_path)
    amp_trg, _ = get_cached_ref_fft(ref_path)

    rows = []
    for beta in betas:
        out_dir = os.path.join(out_base, beta_to_name(beta))
        save_path = safe_out_path(out_dir, rel_name)

        out_img = fda_from_precomputed(amp_src, pha_src, amp_trg, beta=beta)
        save_rgb(out_img, save_path)

        rows.append({
            "original_path": img_path,
            "relative_name": rel_name,
            "reference_image": ref_path,
            "beta": beta,
            "output_path": save_path,
        })

    return rows

# =========================================================
# 7. 单个外部集处理
# =========================================================
def process_one_testset(test_name: str, test_dir: str, style_pool: List[str]):
    print(f"\n🚀 Processing {test_name}: {test_dir}")

    test_imgs = list_images(test_dir, recursive=RECURSIVE)
    if len(test_imgs) == 0:
        print(f"⚠️ {test_name} 没有图像，跳过")
        return

    out_base = os.path.join(OUT_ROOT, test_name)
    ensure_dir(out_base)

    # 预先创建 beta 根目录
    for beta in BETAS:
        ensure_dir(os.path.join(out_base, beta_to_name(beta)))

    # 使用较小参考池，减少随机性和缓存压力
    style_pool_small = choose_small_reference_pool(style_pool, REF_POOL_SIZE)
    print(f"   test images        : {len(test_imgs)}")
    print(f"   reference pool all : {len(style_pool)}")
    print(f"   reference pool use : {len(style_pool_small)}")
    print(f"   workers            : {NUM_WORKERS}")
    print(f"   betas              : {BETAS}")

    tasks = [(img_path, test_dir, out_base, BETAS) for img_path in test_imgs]

    rows = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        initargs=(style_pool_small,),
    ) as ex:
        for result in tqdm(
            ex.map(process_one_image_worker, tasks, chunksize=CHUNK_SIZE),
            total=len(tasks),
            desc=test_name
        ):
            rows.extend(result)

    pd.DataFrame(rows).to_csv(
        os.path.join(out_base, "metadata.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"✅ Done: {test_name}")

# =========================================================
# 8. 主程序
# =========================================================
def main():
    set_seed(SEED)
    ensure_dir(OUT_ROOT)

    print("=" * 80)
    print("FDA batch optimized pipeline")
    print("=" * 80)
    print(f"OUT_ROOT       : {OUT_ROOT}")
    print(f"N_REF_IMAGES   : {N_REF_IMAGES}")
    print(f"REF_POOL_SIZE  : {REF_POOL_SIZE}")
    print(f"BETAS          : {BETAS}")
    print(f"NUM_WORKERS    : {NUM_WORKERS}")
    print(f"CHUNK_SIZE     : {CHUNK_SIZE}")
    print("=" * 80)

    style_pool = build_reference_pool()

    for test_name, test_dir in TEST_DIRS.items():
        process_one_testset(test_name, test_dir, style_pool)

    print("\n🎉 FDA finished.")

if __name__ == "__main__":
    main()