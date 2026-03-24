import json
import cv2
import numpy as np
from pathlib import Path
import os
import math
from tqdm import tqdm

'''
基于金标准（Ground Truth）标注的单细胞图像提取、裁剪和标签标记
替代原YOLO检测流程，直接使用JSON标注文件中的多边形进行分割
'''

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
    
    参数:
        points: [[x1, y1], [x2, y2], ...] 格式的点列表
        offset_x, offset_y: 坐标偏移量（用于转换到局部坐标系）
    """
    if not points or len(points) < 3:
        return None
    
    points_array = np.array([[int(p[0] - offset_x), int(p[1] - offset_y)] for p in points],
                          dtype=np.int32)
    # reshape 成 OpenCV 所需格式：(N, 1, 2)
    return points_array.reshape(-1, 1, 2)


def load_ground_truth_polygons(points_json_path):
    """
    加载真实标注的多边形轮廓和标签
    返回包含轮廓、标签、原始点集的字典列表
    """
    gt_polygons = []
    try:
        with open(points_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data.get('shapes', []):
            # 支持 polygon 和 polyline 类型
            if shape.get('shape_type') in ['polygon', 'polyline'] and shape.get('points'):
                pts = shape['points']
                # 转换为 numpy 数组 (N, 1, 2)
                contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                label = shape.get('label', '0')
                
                # 计算边界框中心（用于裁剪定位）
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                
                gt_polygons.append({
                    'contour': contour,
                    'label': label,
                    'points': pts,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': cv2.boundingRect(contour)  # (x, y, w, h)
                })
    except Exception as e:
        print(f"加载标注文件失败 {points_json_path}: {e}")
        
    return gt_polygons


def is_cell_complete(contour, image_shape, crop_region):
    """
    检查细胞是否完整（不在裁剪区域边缘）
    
    参数:
        contour: 轮廓（局部坐标系）
        image_shape: 裁剪后图像的形状 (h, w)
        crop_region: (x1, y1, x2, y2) 裁剪区域在原图上的坐标
    """
    if contour is None:
        return False
    
    x, y, w, h = cv2.boundingRect(contour)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    margin = 10  # 边缘容差
    
    # 检查轮廓是否接触裁剪区域的边缘
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
    else:
        return False


def process_cells_from_ground_truth(points_json_path, image_path, output_dir, 
                                    remove_background=True, filter_edge_cells=True, 
                                    min_circularity=0.8, min_area=10000, crop_size=600, 
                                    output_size=None):
    """
    基于金标准标注处理单细胞裁剪和标签标记
    
    参数:
        points_json_path: 标注JSON文件路径（包含多边形标注）
        image_path: 原始图像路径
        output_dir: 输出目录
        remove_background: 是否去除背景（仅保留细胞区域）
        filter_edge_cells: 是否过滤边缘不完整的细胞
        min_circularity: 最小圆度阈值
        min_area: 最小面积阈值
        crop_size: 裁剪区域尺寸（正方形）
        output_size: 最终输出图像尺寸（None则等于crop_size）
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在：{image_path}")
        return
    
    image = imread_chinese(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return
    
    # 加载金标准多边形标注
    gt_polygons = load_ground_truth_polygons(points_json_path)
    
    if not gt_polygons:
        print(f"未在标注文件中找到有效多边形：{points_json_path}")
        return
    
    print(f"从标注文件加载了 {len(gt_polygons)} 个细胞")
    
    # 如果未指定输出尺寸，默认与裁剪尺寸一致
    final_size = output_size if output_size is not None else crop_size
    
    valid_cell_count = 0
    
    for i, gt_data in enumerate(gt_polygons):
        # 直接使用金标准的中心点和轮廓
        center_x = gt_data['center_x']
        center_y = gt_data['center_y']
        raw_contour = gt_data['contour']
        cell_label = gt_data['label']
        
        # 计算裁剪区域（以细胞中心为中心的正方形）
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
        
        # 将轮廓转换到裁剪区域内的局部坐标系
        contour_local = raw_contour.copy()
        contour_local[:, :, 0] -= crop_x1_img
        contour_local[:, :, 1] -= crop_y1_img
        
        crop_region = (0, 0, crop_size, crop_size)
        
        # === 筛选流程 ===
        # 1. 检查是否为边缘细胞
        if filter_edge_cells:
            # 检查原始轮廓是否接触图像边界
            orig_x, orig_y, orig_w, orig_h = gt_data['bbox']
            img_h, img_w = image.shape[:2]
            margin = 10
            
            if (orig_x <= margin or orig_x + orig_w >= img_w - margin or 
                orig_y <= margin or orig_y + orig_h >= img_h - margin):
                continue
        
        # 2. 计算圆度和面积（使用局部轮廓）
        area = cv2.contourArea(contour_local)
        perimeter = cv2.arcLength(contour_local, True)
        
        if area < min_area:
            continue
        
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * math.pi * area) / (perimeter * perimeter)
        
        if circularity < min_circularity:
            continue
        
        # === 开始裁剪图像 ===
        # 确保裁剪区域在图像范围内
        crop_x1_img = max(0, crop_x1_img)
        crop_y1_img = max(0, crop_y1_img)
        crop_x2_img = min(image.shape[1], crop_x2_img)
        crop_y2_img = min(image.shape[0], crop_y2_img)
        
        cropped_image = image[crop_y1_img:crop_y2_img, crop_x1_img:crop_x2_img].copy()
        
        # 如果裁剪尺寸不足，进行填充（应对边缘情况）
        if cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            h, w = cropped_image.shape[:2]
            padded[:h, :w] = cropped_image
            cropped_image = padded
            
            # 更新轮廓偏移（如果有填充）
            # 注意：这里假设只在右下方向填充，所以不需要额外偏移调整
        
        # 去除背景（仅保留多边形内的区域）
        if remove_background:
            mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            # 确保轮廓在有效范围内
            contour_clipped = contour_local.copy()
            contour_clipped = np.clip(contour_clipped, 0, crop_size - 1)
            cv2.fillPoly(mask, [contour_clipped], 255)
            cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        
        # === 尺寸调整 ===
        if cropped_image.shape[0] != final_size or cropped_image.shape[1] != final_size:
            cropped_image = cv2.resize(cropped_image, (final_size, final_size))
        
        # 保存文件
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_{valid_cell_count:03d}_{cell_label}.png"
        output_path = output_dir / output_filename
        
        imwrite_chinese(cropped_image, str(output_path))
        valid_cell_count += 1
    
    # print(f"成功提取 {valid_cell_count} 个有效细胞")


def batch_process_directory(points_json_dir, image_dir, output_base_dir, 
                          remove_background=True, filter_edge_cells=True, 
                          min_circularity=0.8, min_area=10000, crop_size=576, 
                          output_size=None):
    """
    批量处理目录（基于金标准标注）
    
    参数:
        points_json_dir: 标注JSON文件目录
        image_dir: 原始图像目录
        output_base_dir: 输出基础目录
        其他参数同 process_cells_from_ground_truth
    """
    
    points_json_dir = Path(points_json_dir)
    image_dir = Path(image_dir)
    output_base_dir = Path(output_base_dir)
    
    # 获取所有JSON标注文件
    json_files = list(points_json_dir.rglob("*.json"))
    
    print(f"找到 {len(json_files)} 个标注文件")
    
    for json_file in tqdm(json_files, desc="处理图像"):
        # 构建对应的图像文件名（与JSON文件同名）
        image_name = json_file.stem
        image_file = None
        
        # 尝试常见图像扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            potential_file = image_dir / json_file.relative_to(points_json_dir).parent / f"{image_name}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        # 如果未找到，尝试在整棵目录树中搜索同名文件
        if image_file is None:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                potential_file = image_dir / f"{image_name}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
        
        # 递归搜索（备选方案）
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
        
        # 构建输出子目录（保持目录结构）
        relative_path = json_file.relative_to(points_json_dir).parent
        output_dir = output_base_dir / relative_path
        
        process_cells_from_ground_truth(
            json_file, image_file, output_dir,
            remove_background, filter_edge_cells, 
            min_circularity, min_area, 
            crop_size, output_size
        )


if __name__ == "__main__":
    # 配置参数
    target_output_size = 576  # 最终输出尺寸（如224用于分类模型）
    
    datasets = [
        {
            "name": "train_groundtruth",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",  # 标注JSON目录
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",   # 原始图像目录
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/train_groundtruth"
        },
        {
            "name": "val_groundtruth",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",  # 标注JSON目录
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",   # 原始图像目录
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/val_groundtruth"
        },
    ]

    for ds in datasets:
        print(f"\n开始处理数据集：{ds['name']}")
        batch_process_directory(
            ds['label'],  # 标注目录（替代原seg参数）
            ds['img'], 
            ds['out'], 
            remove_background=False,      # 是否去除背景
            filter_edge_cells=True,       # 是否过滤边缘细胞
            min_circularity=0.65,         # 最小圆度
            min_area=10000,               # 最小面积
            crop_size=576,                # 裁剪尺寸（用于定位和高分辨率处理）
            output_size=target_output_size # 最终输出尺寸
        )