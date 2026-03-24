import json
import cv2
import numpy as np
from pathlib import Path
import os
import math
from tqdm import tqdm

'''
基于金标准（Ground Truth）标注的单细胞图像提取、裁剪和标签标记
去除0标签版本：过滤掉标签为"0"或"V"的细胞，只保留有效类别
'''

# ===================== 类别配置（不含0）====================
# 定义保留的类别映射（与原YOLO训练配置保持一致）
CELL_DICT_BIG = {
    # 排除 "V": 0, "0": 0
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
}

# 有效标签集合（用于快速查找）
VALID_LABELS = set(CELL_DICT_BIG.keys())
EXCLUDED_LABELS = {"0", "V"}  # 明确排除的标签

# 类别名称映射
CLASS_NAMES = {
    1: "normal",
    2: "abnormal"
}


def calculate_circularity(contour):
    """计算轮廓的圆度"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return circularity


def get_contour_from_points(points, offset_x=0, offset_y=0):
    """
    从点列表生成 OpenCV 轮廓 (contour)
    """
    if not points or len(points) < 3:
        return None
    
    points_array = np.array([[int(p[0] - offset_x), int(p[1] - offset_y)] for p in points],
                          dtype=np.int32)
    return points_array.reshape(-1, 1, 2)


def load_ground_truth_polygons(points_json_path):
    """
    加载真实标注的多边形轮廓和标签（过滤0标签）
    只返回有效类别的细胞
    """
    gt_polygons = []
    skipped_count = {"0": 0, "V": 0, "other": 0}
    
    try:
        with open(points_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data.get('shapes', []):
            # 支持 polygon 和 polyline 类型
            if shape.get('shape_type') not in ['polygon', 'polyline']:
                continue
            
            if not shape.get('points'):
                continue
            
            label = shape.get('label', '0')
            
            # 过滤0标签：跳过 "0" 和 "V"
            if label in EXCLUDED_LABELS:
                skipped_count[label] += 1
                continue
            
            # 检查是否在有效类别列表中
            if label not in VALID_LABELS:
                skipped_count["other"] += 1
                continue
            
            pts = shape['points']
            contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            
            # 计算边界框中心（用于裁剪定位）
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # 获取映射后的类别ID
            class_id = CELL_DICT_BIG[label]
            
            gt_polygons.append({
                'contour': contour,
                'label': label,  # 原始标签
                'class_id': class_id,  # 映射后的类别ID
                'class_name': CLASS_NAMES.get(class_id, "unknown"),
                'points': pts,
                'center_x': center_x,
                'center_y': center_y,
                'bbox': cv2.boundingRect(contour)
            })
            
    except Exception as e:
        print(f"加载标注文件失败 {points_json_path}: {e}")
        
    return gt_polygons, skipped_count


def is_cell_complete(contour, image_shape, crop_region):
    """
    检查细胞是否完整（不在裁剪区域边缘）
    """
    if contour is None:
        return False
    
    x, y, w, h = cv2.boundingRect(contour)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    margin = 10
    
    if (x <= crop_x1 + margin or x + w >= crop_x2 - margin or 
        y <= crop_y1 + margin or y + h >= crop_y2 - margin):
        return False
    
    return True


def imread_chinese(path):
    """支持中文路径的图像读取"""
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_chinese(image, path):
    """支持中文路径的图像保存"""
    is_success, buffer = cv2.imencode(Path(path).suffix, image)
    if is_success:
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    return False


def process_cells_from_ground_truth(points_json_path, image_path, output_dir, 
                                    remove_background=True, filter_edge_cells=True, 
                                    min_circularity=0.65, min_area=10000, crop_size=576, 
                                    output_size=None):
    """
    基于金标准标注处理单细胞裁剪和标签标记（去除0标签版本）
    
    输出文件名格式：{image_name}_{cell_id:03d}_{original_label}.png
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在：{image_path}")
        return 0, 0
    
    image = imread_chinese(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return 0, 0
    
    # 加载金标准多边形标注（自动过滤0标签）
    gt_polygons, skip_stats = load_ground_truth_polygons(points_json_path)
    
    total_cells = len(gt_polygons) + sum(skip_stats.values())
    
    if not gt_polygons:
        return 0, total_cells
    
    final_size = output_size if output_size is not None else crop_size
    
    valid_cell_count = 0
    filtered_stats = {"edge": 0, "area": 0, "circularity": 0}
    
    for gt_data in gt_polygons:
        center_x = gt_data['center_x']
        center_y = gt_data['center_y']
        raw_contour = gt_data['contour']
        original_label = gt_data['label']
        class_id = gt_data['class_id']
        
        # 计算裁剪区域
        crop_x1_img = max(0, int(center_x - crop_size / 2))
        crop_y1_img = max(0, int(center_y - crop_size / 2))
        crop_x2_img = min(image.shape[1], crop_x1_img + crop_size)
        crop_y2_img = min(image.shape[0], crop_y1_img + crop_size)
        
        # 调整以保持固定裁剪尺寸
        if crop_x2_img - crop_x1_img < crop_size:
            if crop_x1_img == 0:
                crop_x2_img = crop_size
            else:
                crop_x1_img = crop_x2_img - crop_size
        if crop_y2_img - crop_y1_img < crop_size:
            if crop_y1_img == 0:
                crop_y2_img = crop_size
            else:
                crop_y1_img = crop_y2_img - crop_size
        
        # 转换到局部坐标系
        contour_local = raw_contour.copy()
        contour_local[:, :, 0] -= crop_x1_img
        contour_local[:, :, 1] -= crop_y1_img
        
        crop_region = (0, 0, crop_size, crop_size)
        
        # === 筛选流程 ===
        # 1. 边缘检查
        if filter_edge_cells:
            orig_x, orig_y, orig_w, orig_h = gt_data['bbox']
            img_h, img_w = image.shape[:2]
            margin = 10
            
            if (orig_x <= margin or orig_x + orig_w >= img_w - margin or 
                orig_y <= margin or orig_y + orig_h >= img_h - margin):
                filtered_stats["edge"] += 1
                continue
        
        # 2. 面积筛选
        area = cv2.contourArea(contour_local)
        if area < min_area:
            filtered_stats["area"] += 1
            continue
        
        # 3. 圆度筛选
        perimeter = cv2.arcLength(contour_local, True)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * math.pi * area) / (perimeter * perimeter)
        
        if circularity < min_circularity:
            filtered_stats["circularity"] += 1
            continue
        
        # === 裁剪图像 ===
        crop_x1_img = max(0, crop_x1_img)
        crop_y1_img = max(0, crop_y1_img)
        crop_x2_img = min(image.shape[1], crop_x2_img)
        crop_y2_img = min(image.shape[0], crop_y2_img)
        
        cropped_image = image[crop_y1_img:crop_y2_img, crop_x1_img:crop_x2_img].copy()
        
        # 填充到固定尺寸
        if cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            h, w = cropped_image.shape[:2]
            padded[:h, :w] = cropped_image
            cropped_image = padded
        
        # 去除背景
        if remove_background:
            mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            contour_clipped = np.clip(contour_local, 0, crop_size - 1)
            cv2.fillPoly(mask, [contour_clipped], 255)
            cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        
        # 尺寸调整
        if cropped_image.shape[0] != final_size or cropped_image.shape[1] != final_size:
            cropped_image = cv2.resize(cropped_image, (final_size, final_size))
        
        # 保存文件（文件名包含原始标签和映射后的类别ID）
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_{valid_cell_count:03d}_{original_label}.png"
        output_path = output_dir / output_filename
        
        imwrite_chinese(cropped_image, str(output_path))
        valid_cell_count += 1
    
    return valid_cell_count, total_cells


def batch_process_directory(points_json_dir, image_dir, output_base_dir, 
                          remove_background=True, filter_edge_cells=True, 
                          min_circularity=0.65, min_area=10000, crop_size=576, 
                          output_size=None):
    """
    批量处理目录（基于金标准标注，去除0标签）
    """
    
    points_json_dir = Path(points_json_dir)
    image_dir = Path(image_dir)
    output_base_dir = Path(output_base_dir)
    
    json_files = list(points_json_dir.rglob("*.json"))
    
    print(f"找到 {len(json_files)} 个标注文件")
    print(f"保留类别: {list(VALID_LABELS)}")
    print(f"排除类别: {list(EXCLUDED_LABELS)}")
    
    total_stats = {"processed": 0, "valid_cells": 0, "total_cells": 0}
    
    for json_file in tqdm(json_files, desc="处理图像"):
        # 跳过Jupyter备份文件
        if ".ipynb_checkpoints" in str(json_file):
            continue
            
        image_name = json_file.stem
        image_file = None
        
        # 查找对应图像
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            potential_file = image_dir / json_file.relative_to(points_json_dir).parent / f"{image_name}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        if image_file is None:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                potential_file = image_dir / f"{image_name}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
        
        if image_file is None:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                for img_file in image_dir.rglob(f"*{ext}"):
                    if img_file.stem == image_name:
                        image_file = img_file
                        break
                if image_file:
                    break
        
        if image_file is None:
            print(f"未找到对应的图像文件：{image_name}")
            continue
        
        # 构建输出子目录
        relative_path = json_file.relative_to(points_json_dir).parent
        output_dir = output_base_dir / relative_path
        
        valid, total = process_cells_from_ground_truth(
            json_file, image_file, output_dir,
            remove_background, filter_edge_cells, 
            min_circularity, min_area, 
            crop_size, output_size
        )
        
        if total > 0:
            total_stats["processed"] += 1
            total_stats["valid_cells"] += valid
            total_stats["total_cells"] += total
    
    print(f"\n📊 批量处理统计：")
    print(f"  - 成功处理文件数: {total_stats['processed']}/{len(json_files)}")
    print(f"  - 总细胞数（含0标签）: {total_stats['total_cells']}")
    print(f"  - 有效细胞数（不含0）: {total_stats['valid_cells']}")
    if total_stats['total_cells'] > 0:
        ratio = total_stats['valid_cells'] / total_stats['total_cells'] * 100
        print(f"  - 保留比例: {ratio:.1f}%")


if __name__ == "__main__":
    # 配置参数
    target_output_size = 576
    
    datasets = [
        {
            "name": "train_without0",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/train_groundtruth_without0"
        },
        {
            "name": "val_without0",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/val_groundtruth_without0"
        },
        # {
        #     "name": "test_BEPH",
        #     "label": r"/root/autodl-tmp/data/BEPH_imgs_260211",
        #     "img": r"/root/autodl-tmp/data/BEPH_imgs_260211",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/singlecell_gt/test_BEPH"
        # },
        # {
        #     "name": "test_BJH",
        #     "label": r"/root/autodl-tmp/data/BJH_imgs_260211",
        #     "img": r"/root/autodl-tmp/data/BJH_imgs_260211",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/singlecell_gt/test_BJH"
        # },
        # {
        #     "name": "test_FXH",
        #     "label": r"/root/autodl-tmp/data/FXH_imgs_260318",
        #     "img": r"/root/autodl-tmp/data/FXH_imgs_260318",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/singlecell_gt/test_FXH"
        # },
        # {
        #     "name": "test_TJMU",
        #     "label": r"/root/autodl-tmp/data/TJMU_imgs_260318",
        #     "img": r"/root/autodl-tmp/data/TJMU_imgs_260318",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/singlecell_gt/test_TJMU"
        # },
        # {
        #     "name": "test_FXH_noALL",
        #     "label": r"/root/autodl-tmp/data/FXH_imgs_noALL_260318",
        #     "img": r"/root/autodl-tmp/data/FXH_imgs_noALL_260318",
        #     "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/singlecell_gt/test_FXH_noALL"
        # },
    ]

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"开始处理数据集：{ds['name']}")
        print(f"{'='*50}")
        
        batch_process_directory(
            ds['label'],
            ds['img'], 
            ds['out'], 
            remove_background=False,
            filter_edge_cells=True,
            min_circularity=0.65,
            min_area=10000,
            crop_size=576,
            output_size=target_output_size
        )