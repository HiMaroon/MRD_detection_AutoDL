# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import time

# ============================== 辅助函数 ==============================
def is_image_file(filename):
    """判断文件是否为图片格式"""
    ext = filename.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']

def get_all_image_files(src_dir):
    """递归获取目录下所有图片文件（包括子目录）"""
    src_path = Path(src_dir)
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
        image_files.extend(src_path.rglob(f"*.{ext}"))
        image_files.extend(src_path.rglob(f"*.{ext.upper()}"))
    return image_files

# ============================== 类别映射 ==============================
cell_dict_big = {
    "V": 0, "0": 0,
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
}

cell_dict_small = {
    "V": 0, "0": 0, "N": 1, "N1": 1, "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2, "M": 3, "M1": 3, "M0": 4, "M2": 4,
    "R": 5, "R1": 5, "R2": 6, "R3": 6, "J": 7, "J1": 7, "J2": 8, "J3": 8, "J4": 8,
    "L": 9, "L1": 9, "L2": 9, "L3": 9, "L4": 9, "P": 10, "P1": 10, "P2": 10, "P3": 10
}

class MyRotateTransform:
    def __init__(self, angles): 
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

transCrop = 576

# ============================== 单张图片增强函数（用于并行处理） ==============================
def augment_single_image(args):
    """处理单张图片的增强任务"""
    src_path, dst_base, rel_path, aug_num = args
    
    try:
        # 每个进程独立创建transform实例（避免共享问题）
        transformSequence = transforms.Compose([
            transforms.Resize([transCrop, transCrop]),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
        ])
        
        img = Image.open(src_path).convert('RGB')
        name_no_ext = os.path.splitext(rel_path.name)[0]
        
        # 保持子目录结构
        dst_dir = dst_base / rel_path.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量生成增强图像（统一保存为 jpg 格式）
        for i in range(aug_num):
            aug_img = transformSequence(img)
            save_name = f"{name_no_ext}_{i}.jpg"
            save_path = dst_dir / save_name
            aug_img.save(str(save_path), quality=95)
        
        return str(rel_path), True, None
    except Exception as e:
        return str(rel_path), False, str(e)

# ============================== 批处理增强函数 ==============================
def DataPicGenerator_batch(src_dir, dst_dir, aug_num=5, num_workers=None):
    """
    批处理版本的数据增强函数（支持子目录）
    """
    src_path = Path(src_dir)
    dst_base = Path(dst_dir)
    
    if not dst_base.exists():
        dst_base.mkdir(parents=True, exist_ok=True)
    
    # 递归获取所有图片文件（支持多种格式 + 子目录）
    file_list = get_all_image_files(src_path)
    
    if not file_list:
        print(f"⚠️ 警告：目录 {src_dir} 中没有找到图片文件")
        return
    
    # 设置默认工作进程数
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 100)  # 保留一个CPU给系统
    
    print(f"🚀 开始处理目录：{src_path.name}")
    print(f"📊 总图片数：{len(file_list)}，使用 {num_workers} 个进程并行处理")
    
    # 准备任务参数（包含相对路径以保持目录结构）
    task_args = [(str(f), dst_base, f.relative_to(src_path), aug_num) for f in file_list]
    
    # 使用进程池并行处理
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(augment_single_image, args): args[2] 
                         for args in task_args}
        
        # 使用tqdm显示进度
        with tqdm(total=len(file_list), desc=f"增强 {src_path.name}", 
                  unit="img", ncols=100) as pbar:
            for future in as_completed(future_to_file):
                rel_path, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    # 可以取消下面注释以查看错误详情
                    # tqdm.write(f"❌ 处理失败 {rel_path}: {error_msg}")
                
                pbar.update(1)
                pbar.set_postfix({"成功": success_count, "失败": fail_count})
    
    print(f"✅ 目录 {src_path.name} 处理完成：成功 {success_count} / 失败 {fail_count}")

# ============================== 批处理标签生成函数 ==============================
def DataTxtGenerator_batch(output_dir, num_workers=None):
    """
    批处理版本的标签文件生成函数（支持子目录）
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 100)
    
    output_path = Path(output_dir)
    
    for split_name in ['train', 'val']:
        img_dir = output_path / split_name
        if not img_dir.exists():
            print(f"⚠️ 跳过：找不到目录 {img_dir}")
            continue
        
        # 递归获取所有图片（支持多种格式 + 子目录）
        imgs = get_all_image_files(img_dir)
        if not imgs:
            print(f"⚠️ 目录 {img_dir} 中没有图片文件")
            continue
        
        txt_path = output_path / f"{split_name}_labels.txt"
        
        print(f"📝 生成 {split_name} 标签文件，使用 {num_workers} 个线程")
        
        # 准备数据
        label_data = []
        
        # 使用线程池并行处理标签提取（IO密集型任务）
        def extract_label(img_path):
            # img_path 是 Path 对象
            rel_path = img_path.relative_to(img_dir)
            filename = img_path.name
            
            # 从文件名提取标签（格式：name_label.jpg）
            base_name = filename.rsplit('_', 1)[0]
            parts = base_name.split('_')
            raw_label = parts[-1]
            
            label_big = cell_dict_big.get(raw_label, 0)
            label_small = cell_dict_small.get(raw_label, 0)
            
            # 标签文件中包含相对路径（便于追溯）
            return f"{img_path} {label_big} {label_small}\n"
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_label, img_path) for img_path in imgs]
            
            with tqdm(total=len(imgs), desc=f"生成 {split_name} 标签", 
                      unit="line", ncols=80) as pbar:
                for future in as_completed(futures):
                    label_data.append(future.result())
                    pbar.update(1)
        
        # 写入文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(label_data)
        
        print(f"✅ {split_name} 标签文件生成完成：{txt_path}")

# ============================== 内存优化版本的增强函数（适合大文件） ==============================
def DataPicGenerator_chunked(src_dir, dst_dir, aug_num=5, chunk_size=100):
    """
    分块处理版本，避免内存占用过高（支持子目录）
    """
    src_path = Path(src_dir)
    dst_base = Path(dst_dir)
    
    if not dst_base.exists():
        dst_base.mkdir(parents=True, exist_ok=True)
    
    # 递归获取所有图片文件（支持多种格式 + 子目录）
    file_list = get_all_image_files(src_path)
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 将文件列表分块
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    
    print(f"🚀 开始处理目录：{src_path.name}")
    print(f"📊 总图片数：{len(file_list)}，分 {len(chunks)} 块处理，每块最多 {chunk_size} 张")
    
    total_success = 0
    total_fail = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"\n📦 处理第 {chunk_idx + 1}/{len(chunks)} 块...")
        
        task_args = [(str(f), dst_base, f.relative_to(src_path), aug_num) for f in chunk]
        
        success_count = 0
        fail_count = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(augment_single_image, args): args[2] 
                             for args in task_args}
            
            with tqdm(total=len(chunk), desc=f"块 {chunk_idx + 1}", 
                      unit="img", ncols=100) as pbar:
                for future in as_completed(future_to_file):
                    rel_path, success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"成功": success_count, "失败": fail_count})
        
        total_success += success_count
        total_fail += fail_count
        print(f"✅ 块 {chunk_idx + 1} 完成：成功 {success_count} / 失败 {fail_count}")
    
    print(f"\n✅ 目录 {src_path.name} 全部完成：总成功 {total_success} / 总失败 {total_fail}")

# ============================== 主程序 ==============================
if __name__ == "__main__":
    pathTrain = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train"
    pathVal   = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val"    
    outputdir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/aug10_260323"
    
    start_time = time.time()
    
    # 选择增强方式：
    # 方式1: 标准批处理（推荐，平衡性能和内存）
    DataPicGenerator_batch(pathTrain, os.path.join(outputdir, 'train'), aug_num=10)
    DataPicGenerator_batch(pathVal, os.path.join(outputdir, 'val'), aug_num=10)
    
    # 方式2: 分块处理（适合超大文件集，内存占用更低）
    # DataPicGenerator_chunked(pathTrain, os.path.join(outputdir, 'train'), aug_num=5, chunk_size=200)
    # DataPicGenerator_chunked(pathVal, os.path.join(outputdir, 'val'), aug_num=5, chunk_size=200)
    
    # 生成标签
    DataTxtGenerator_batch(outputdir)
    
    elapsed_time = time.time() - start_time
    print(f"\n✨ 全部任务已完成！总耗时：{elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")