import json
import cv2
import numpy as np
from pathlib import Path
import os
import math
from tqdm import tqdm

'''
从 yolo 结果提取单细胞图像，并进行裁剪和标签标记
修改版：支持基于轮廓重合度 (IoU) 的标签匹配 + 支持独立设置输出图片尺寸
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

def calculate_contour_iou(contour1, contour2):
    """
    计算两个轮廓的 IoU (Intersection over Union)
    使用掩码计算法，适用于任意形状
    """
    if contour1 is None or contour2 is None:
        return 0.0
    
    # 获取两个轮廓的边界框
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # 计算包含两个轮廓的最小外接矩形
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return 0.0
    
    # 创建局部掩码
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    # 将轮廓坐标平移到局部坐标系
    contour1_shifted = contour1 - np.array([[min_x, min_y]])
    contour2_shifted = contour2 - np.array([[min_x, min_y]])
    
    # 绘制填充掩码
    cv2.fillPoly(mask1, [contour1_shifted], 1)
    cv2.fillPoly(mask2, [contour2_shifted], 1)
    
    # 计算交集和并集
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    
    inter_area = np.sum(intersection)
    union_area = np.sum(union)
    
    if union_area == 0:
        return 0.0
        
    return inter_area / union_area

def get_contour_from_segments(cell_data, offset_x=0, offset_y=0):
    """
    从 cell_data 的 segments 字段生成 OpenCV 轮廓 (contour)
    """
    if ('segments' in cell_data and 
        'x' in cell_data['segments'] and 
        'y' in cell_data['segments']):
        
        x_coords = cell_data['segments']['x']
        y_coords = cell_data['segments']['y']
        
        points = np.array([[int(x - offset_x), int(y - offset_y)] for x, y in zip(x_coords, y_coords)],
                          dtype=np.int32)
        # reshape 成 OpenCV 所需格式：(N, 1, 2)
        return points.reshape(-1, 1, 2)
    
    return None

def load_ground_truth_polygons(points_json_path):
    """
    加载真实标注的多边形轮廓和标签
    """
    gt_polygons = []
    try:
        with open(points_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data.get('shapes', []):
            # 修改：支持 polygon 类型，不再仅限于 point
            if shape.get('shape_type') in ['polygon', 'polyline'] and shape.get('points'):
                pts = shape['points']
                # 转换为 numpy 数组 (N, 1, 2)
                contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                label = shape.get('label', '0')
                gt_polygons.append({
                    'contour': contour,
                    'label': label
                })
    except Exception as e:
        print(f"加载标注文件失败 {points_json_path}: {e}")
        
    return gt_polygons

def is_cell_complete(contour, image_shape, crop_region):
    """
    检查细胞是否完整（不在图像边缘）
    """
    if contour is None:
        return False
    
    x, y, w, h = cv2.boundingRect(contour)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    margin = 10  # 边缘容差
    
    if (x <= crop_x1 + margin or x + w >= crop_x2 - margin or 
        y <= crop_y1 + margin or y + h >= crop_y2 - margin):
        return False
    
    return True

def imread_chinese(path):
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_chinese(image, path):
    is_success, buffer = cv2.imencode(Path(path).suffix, image)
    if is_success:
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    else:
        return False

def process_cells(segmentation_json_path, points_json_path, image_path, output_dir, 
                 remove_background=True, filter_edge_cells=True, 
                 min_circularity=0.8, min_area=10000, crop_size=600, 
                 output_size=None, iou_threshold=0.5):
    """
    处理单细胞裁剪和标签标记 (基于轮廓 IoU 匹配)
    
    参数:
    crop_size: 用于裁剪、筛选、掩码制作的原始尺寸 (保持高分辨率用于筛选)
    output_size: 最终保存的图片尺寸 (用于模型训练，若为 None 则默认等于 crop_size)
    iou_threshold: 检测轮廓与真实轮廓重合度阈值，超过此值才赋予标签
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
    
    with open(segmentation_json_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    
    # 加载真实标注的多边形 (Ground Truth)
    gt_polygons = load_ground_truth_polygons(points_json_path)
    
    # 如果未指定输出尺寸，默认与裁剪尺寸一致
    final_size = output_size if output_size is not None else crop_size
    
    valid_cell_count = 0
    for i, cell_data in enumerate(seg_data):
        box = cell_data.get('box', {})
        x1, y1, x2, y2 = (
            box.get('x1', 0),
            box.get('y1', 0),
            box.get('x2', 0),
            box.get('y2', 0)
        )

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        
        crop_x1_img = max(0, int(center_x - crop_size / 2))
        crop_y1_img = max(0, int(center_y - crop_size / 2))
        crop_x2_img = min(image.shape[1], crop_x1_img + crop_size)
        crop_y2_img = min(image.shape[0], crop_y1_img + crop_size)

        # 调整以保持固定尺寸 (基于 crop_size)
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

        # 获取全局坐标系下的检测轮廓
        raw_contour = get_contour_from_segments(cell_data)
        if raw_contour is None:
            continue

        # === 标签匹配逻辑 (基于 IoU) ===
        cell_label = '0'
        max_iou = 0.0
        
        for gt in gt_polygons:
            gt_contour = gt['contour']
            # 计算检测轮廓与真实轮廓的 IoU
            iou = calculate_contour_iou(raw_contour, gt_contour)
            if iou > max_iou:
                max_iou = iou
                cell_label = gt['label']
        
        # 如果最大重合度未达到阈值，视为未知或背景
        if max_iou < iou_threshold:
            cell_label = '0' 
            # 如果希望完全跳过未匹配到标签的细胞，可在此处 continue
            # continue 

        # === 筛选流程 ===
        # 将原始轮廓转换到裁剪区域内的局部坐标系 (用于边缘检查和掩码)
        # 注意：这里依然基于 crop_size 进行计算，保证筛选准确性
        contour_local = raw_contour.copy()
        contour_local[:, :, 0] -= crop_x1_img
        contour_local[:, :, 1] -= crop_y1_img

        crop_region = (0, 0, crop_size, crop_size)
        
        # 1. 检查是否为边缘细胞
        if filter_edge_cells:
            if not is_cell_complete(contour_local, (crop_size, crop_size), crop_region):
                continue
        # 2. 计算圆度和面积 (使用局部轮廓，基于 crop_size 分辨率)
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
        cropped_image = image[crop_y1_img:crop_y2_img, crop_x1_img:crop_x2_img].copy()

        if remove_background:
            mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour_local], 255)
            cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

        # === 尺寸调整 (新增逻辑) ===
        # 确保输出大小符合 output_size 设定
        # 注意：之前的逻辑是强制等于 crop_size，现在改为等于 final_size
        if cropped_image.shape[0] != final_size or cropped_image.shape[1] != final_size:
            cropped_image = cv2.resize(cropped_image, (final_size, final_size))

        # 保存文件
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_{valid_cell_count:03d}_{cell_label}.png"
        output_path = output_dir / output_filename

        imwrite_chinese(cropped_image, str(output_path))
        valid_cell_count += 1


def batch_process_directory(seg_json_dir, points_json_dir, image_dir, output_base_dir, 
                          remove_background=True, filter_edge_cells=True, 
                          min_circularity=0.8, min_area=10000, crop_size=576, 
                          output_size=None, iou_threshold=0.5):
    """
    批量处理目录
    """
    
    seg_json_dir = Path(seg_json_dir)
    points_json_dir = Path(points_json_dir)
    image_dir = Path(image_dir)
    output_base_dir = Path(output_base_dir)
    
    seg_json_files = list(seg_json_dir.rglob("*.json"))
    
    for seg_json_file in tqdm(seg_json_files, desc="处理图像"):
        # 构建对应的点标记 JSON 文件名
        points_json_file = points_json_dir / seg_json_file.relative_to(seg_json_dir)
        if not points_json_file.exists():
            points_json_file = points_json_dir / seg_json_file.name
        
        # 构建对应的图像文件名
        image_name = seg_json_file.stem
        image_file = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            potential_file = image_dir / f"{image_name}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        if image_file is None:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                for img_file in image_dir.rglob(f"*{ext}"):
                    if img_file.stem == image_name:
                        image_file = img_file
                        break
                if image_file:
                    break
        
        if image_file is None:
            print(f"未找到对应的图像文件：{image_name}")
            continue
        
        if not points_json_file.exists():
            print(f"未找到对应的标注 JSON 文件：{points_json_file}")
            continue
        
        process_cells(seg_json_file, points_json_file, image_file, output_base_dir,
                     remove_background, filter_edge_cells, min_circularity, min_area, 
                     crop_size, output_size, iou_threshold)

if __name__ == "__main__":
    # 示例配置
    iou_thresh = 0.5 
    
    # 新增配置项：output_size
    # 如果 output_size 为 None，则保存尺寸等于 crop_size
    # 如果 output_size 为 224，则裁剪 600x600 用于筛选，最后缩放为 224x224 保存
    target_output_size = 576  # 例如：设置为 224 用于分类模型输入
    
    datasets = [
        {
            "name": "train",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/train",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train"
        },
        {
            "name": "val",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/val",
            "label": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",
            "img": r"/root/autodl-tmp/data/MAIN_imgs_split_260312/Val",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val"
        },
        {
            "name": "test_BJH",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_BJH",
            "label": r"/root/autodl-tmp/data/BJH_imgs_260211",
            "img": r"/root/autodl-tmp/data/BJH_imgs_260211",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH"
        },
        {
            "name": "test_TJMU",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_TJMU",
            "label": r"/root/autodl-tmp/data/TJMU_imgs_260318",
            "img": r"/root/autodl-tmp/data/TJMU_imgs_260318",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU"
        },
        {
            "name": "test_FXH_noALL",
            "seg": r"/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/test_FXH_noALL",
            "label": r"/root/autodl-tmp/data/FXH_imgs_noALL_260318",
            "img": r"/root/autodl-tmp/data/FXH_imgs_noALL_260318",
            "out": r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL"
        },
    ]

    for ds in datasets:
        print(f"开始处理数据集：{ds['name']}")
        batch_process_directory(
            ds['seg'], ds['label'], ds['img'], ds['out'], 
            remove_background=False, 
            filter_edge_cells=True, 
            min_circularity=0.65, 
            min_area=10000,
            crop_size=576,          # 用于定位、筛选、去背景的原始高分辨率尺寸
            output_size=target_output_size, # 【新增】最终保存图片的尺寸
            iou_threshold=iou_thresh
        )