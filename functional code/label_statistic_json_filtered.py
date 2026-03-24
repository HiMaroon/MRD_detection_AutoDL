import os
import json
import collections
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import math

'''
统计每个指定文件夹的总标签分布（汇总该文件夹下所有子文件夹）
修改版：加入细胞筛选逻辑（边缘、面积、圆度、裁剪尺寸），统计有效细胞分布
细化版：平行统计四种筛选原因的剔除情况
增强版：统计每个细胞类型因四种原因被筛除的数目（主目录级别）
'''

# ================= 配置区域 =================
# 筛选阈值 (与裁剪脚本保持一致)
MIN_AREA = 10000          # 最小面积
MIN_CIRCULARITY = 0.65     # 最小圆度
EDGE_MARGIN = 10          # 边缘容差像素
FILTER_EDGE_CELLS = True  # 是否启用边缘细胞过滤
CROP_SIZE = 576           # 裁剪尺寸，细胞边界框超过此尺寸则筛除

# 指定要遍历的目录
directories = [
    "/root/autodl-tmp/data/MAIN_imgs_260312"
    # "/root/autodl-tmp/data/FXH_imgs_260211",
    # "/root/autodl-tmp/data/BJH_imgs_260211",
    # "/root/autodl-tmp/data/BEPH_imgs_260211",
    # "/root/autodl-tmp/data/MAIN_imgs_outline_mask_260211",
    # "/root/autodl-tmp/data/MAIN_imgs_dotnoutline_mask_260211",
]

GLOBAL_OUTPUT_ROOT = "/root/autodl-tmp/data/statistics_plots_filtered_0.65_crop576"
os.makedirs(GLOBAL_OUTPUT_ROOT, exist_ok=True)

# 原始标签顺序定义
cell_dict = {
    "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
    "E": 8, "B": 9, "M0": 10, "M": 11, "M1": 12, "M2": 13,
    "R": 14, "R1": 15, "R2": 16, "R3": 17,
    "J": 18, "J1": 19, "J2": 20, "J3": 21, "J4": 22,
    "L": 23, "L1": 24, "L2": 25, "L3": 26, "L4": 27,
    "P": 28, "P1": 29, "P2": 30, "P3": 31,
    "B1": 32, "E1": 33, "A": 34, "F": 35, "V": 36, "0": 36
}

# 按照特定顺序排序的标签列表
sorted_labels = sorted(cell_dict.keys(), key=lambda x: cell_dict[x])

# 大类映射字典
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

# ================= 辅助函数 =================

def calculate_circularity(contour):
    """计算轮廓的圆度"""
    try:
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return (4 * math.pi * area) / (perimeter * perimeter)
    except Exception:
        return 0

def get_image_shape(image_path, cache):
    """获取图像尺寸，带缓存"""
    if image_path in cache:
        return cache[image_path]
    
    if not os.path.exists(image_path):
        return None
    
    # 屏蔽 OpenCV JPEG 警告
    import os as _os
    _os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    cache[image_path] = (h, w)
    return (h, w)

def find_matching_image(json_path, image_dir_cache):
    """根据 JSON 路径寻找对应的图像文件"""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    parent_dir = os.path.dirname(json_path)
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG']:
        potential_path = os.path.join(parent_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None

def is_valid_cell(points, image_shape, filter_edge=True, margin=10, min_area=10000, min_circ=0.7, crop_size=600):
    """
    检查细胞是否满足所有筛选条件
    新增：裁剪尺寸检查 - 细胞边界框在x或y方向超过crop_size则筛除
    Returns:
        (is_valid: bool, reject_reason: str or None)
        reject_reason: None | "area" | "circularity" | "edge" | "crop_size"
    """
    if not points or image_shape is None:
        return False, "invalid_input"
    
    try:
        contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        contour = np.ascontiguousarray(contour)

        # 1. 面积检查
        area = cv2.contourArea(contour)
        if area < min_area:
            return False, "area"
        
        # 2. 圆度检查
        circ = calculate_circularity(contour)
        if circ < min_circ:
            return False, "circularity"
        
        # 3. 边缘检查
        if filter_edge:
            h, w = image_shape
            x, y, bw, bh = cv2.boundingRect(contour)
            if (x <= margin or 
                x + bw >= w - margin or 
                y <= margin or 
                y + bh >= h - margin):
                return False, "edge"
        
        # 4. 裁剪尺寸检查 - 新增
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw >= crop_size or bh >= crop_size:
            return False, "crop_size"
        
        return True, None
    except Exception as e:
        return False, "exception"

def calculate_category_stats(label_counts, cell_dict_big):
    """根据 cell_dict_big 映射计算各类别统计"""
    cat0_count = 0
    cat1_count = 0
    cat2_count = 0
    undefined_count = 0
    
    for label, count in label_counts.items():
        category = cell_dict_big.get(label, -1)
        if category == 0:
            cat0_count += count
        elif category == 1:
            cat1_count += count
        elif category == 2:
            cat2_count += count
        else:
            undefined_count += count
    
    total_all = cat0_count + cat1_count + cat2_count + undefined_count
    percentage = (cat1_count / total_all * 100) if total_all > 0 else 0.0
    
    return cat0_count, cat1_count, cat2_count, undefined_count, percentage, total_all

def calculate_reject_stats(reject_counts, total_raw):
    """计算剔除统计的百分比"""
    stats = {}
    for reason in ["area", "circularity", "edge", "crop_size", "invalid_input", "exception"]:
        count = reject_counts.get(reason, 0)
        pct = (count / total_raw * 100) if total_raw > 0 else 0
        stats[reason] = {"count": count, "percentage": pct}
    return stats

def calculate_reject_by_label_stats(reject_counts_by_label, label_raw_counts):
    """
    计算每个标签的剔除原因统计
    Returns:
        dict: {label: {reason: {"count": int, "percentage": float}}}
    """
    stats = {}
    for label in sorted_labels:
        raw_count = label_raw_counts.get(label, 0)
        label_rejects = reject_counts_by_label.get(label, {})
        
        stats[label] = {
            "raw_count": raw_count,
            "rejects": {}
        }
        
        for reason in ["area", "circularity", "edge", "crop_size"]:
            count = label_rejects.get(reason, 0)
            pct = (count / raw_count * 100) if raw_count > 0 else 0
            stats[label]["rejects"][reason] = {
                "count": count,
                "percentage": pct
            }
    
    return stats

def write_detailed_csv(output_path, dataset_name, label_counts, dir_pic_counts, dir_subfolder_counts, 
                       cat0, cat1, cat2, undefined_count, cat1_pct, total_all, 
                       reject_stats, total_raw_annotations,
                       reject_by_label_stats, label_raw_counts,
                       subfolder_summary_list=None, filter_info=None):
    """写入详细的 CSV 统计文件（含剔除原因统计）"""
    with open(output_path, "w", encoding="utf-8-sig", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 1. 文件头信息
        writer.writerow(["Dataset", dataset_name])
        if filter_info:
            writer.writerow(["Filter Settings", f"Area>={filter_info['area']}, Circ>={filter_info['circ']}, Margin={filter_info['margin']}, CropSize<={filter_info['crop_size']}"])
        writer.writerow(["Total Images", dir_pic_counts])
        writer.writerow(["Total Subfolders", dir_subfolder_counts])
        writer.writerow(["Total Raw Annotations", total_raw_annotations])
        writer.writerow(["Total Valid Cells", total_all])
        writer.writerow(["Total Rejected Cells", total_raw_annotations - total_all])
        writer.writerow([])
        
        # 2. 剔除原因平行统计
        writer.writerow(["=== Rejection Reasons (Parallel Statistics) ==="])
        writer.writerow(["Reason", "Description", "Count", "Percentage of Raw"])
        reason_desc = {
            "area": f"Area < {filter_info['area'] if filter_info else MIN_AREA}",
            "circularity": f"Circularity < {filter_info['circ'] if filter_info else MIN_CIRCULARITY}",
            "edge": f"Within {filter_info['margin'] if filter_info else EDGE_MARGIN}px of image border",
            "crop_size": f"Bounding box exceeds {filter_info['crop_size'] if filter_info else CROP_SIZE}px in width or height",
            "invalid_input": "Invalid points or missing image",
            "exception": "Processing exception"
        }
        for reason in ["area", "circularity", "edge", "crop_size", "invalid_input", "exception"]:
            if reason in reject_stats:
                s = reject_stats[reason]
                writer.writerow([reason, reason_desc.get(reason, ""), s["count"], f"{s['percentage']:.2f}%"])
        writer.writerow([])
        
        # 3. 【新增】各标签剔除原因明细表
        writer.writerow(["=== Rejection by Cell Type (Label-Specific) ==="])
        writer.writerow(["Label", "Raw Count", "Valid Count", "Rejected", 
                        "Reject(Area)", "Reject(Circ)", "Reject(Edge)", "Reject(CropSize)",
                        "Area%", "Circ%", "Edge%", "CropSize%"])
        
        for label in sorted_labels:
            raw_count = label_raw_counts.get(label, 0)
            valid_count = label_counts.get(label, 0)
            rejected = raw_count - valid_count
            
            if raw_count > 0:
                label_stats = reject_by_label_stats.get(label, {})
                area_reject = label_stats.get("rejects", {}).get("area", {}).get("count", 0)
                circ_reject = label_stats.get("rejects", {}).get("circularity", {}).get("count", 0)
                edge_reject = label_stats.get("rejects", {}).get("edge", {}).get("count", 0)
                crop_reject = label_stats.get("rejects", {}).get("crop_size", {}).get("count", 0)
                
                area_pct = label_stats.get("rejects", {}).get("area", {}).get("percentage", 0)
                circ_pct = label_stats.get("rejects", {}).get("circularity", {}).get("percentage", 0)
                edge_pct = label_stats.get("rejects", {}).get("edge", {}).get("percentage", 0)
                crop_pct = label_stats.get("rejects", {}).get("crop_size", {}).get("percentage", 0)
                
                writer.writerow([
                    label, raw_count, valid_count, rejected,
                    area_reject, circ_reject, edge_reject, crop_reject,
                    f"{area_pct:.2f}%", f"{circ_pct:.2f}%", f"{edge_pct:.2f}%", f"{crop_pct:.2f}%"
                ])
        writer.writerow([])
        
        # 4. 类别统计汇总
        writer.writerow(["=== Category Statistics (by cell_dict_big) ==="])
        writer.writerow(["Category", "Description", "Count", "Percentage of Valid Cells"])
        
        total_for_pct = total_all if total_all > 0 else 1
        writer.writerow(["0", "Background (V, 0)", cat0, f"{cat0/total_for_pct*100:.2f}%"])
        writer.writerow(["1", "Target Cells (N, N1, M, M1, R, R1, J, J1)", cat1, f"{cat1_pct:.2f}%"])
        writer.writerow(["2", "Other Cells", cat2, f"{cat2/total_for_pct*100:.2f}%"])
        
        if undefined_count > 0:
            writer.writerow(["Undefined", "Labels not in cell_dict_big (Total)", undefined_count, f"{undefined_count/total_for_pct*100:.2f}%"])
            writer.writerow(["--- Undefined Label Details ---", "", "", ""])
            undefined_labels = {k: v for k, v in label_counts.items() if k not in cell_dict_big}
            sorted_undefined = sorted(undefined_labels.keys())
            for u_label in sorted_undefined:
                u_count = undefined_labels[u_label]
                u_pct = (u_count / total_all * 100) if total_all > 0 else 0
                writer.writerow(["", f"  -> {u_label}", u_count, f"{u_pct:.2f}%"])
        writer.writerow([])
        
        # 5. 各标签详细统计 (已知标签)
        writer.writerow(["=== Label Details (Defined) ==="])
        writer.writerow(["Label", "Order", "BigCategory", "Count", "Percentage of Valid"])
        
        for label in sorted_labels:
            count = label_counts.get(label, 0)
            if count > 0:
                pct_total = (count / total_all * 100) if total_all > 0 else 0
                big_cat = cell_dict_big.get(label, "N/A")
                writer.writerow([
                    label, 
                    cell_dict[label], 
                    big_cat,
                    count, 
                    f"{pct_total:.2f}%"
                ])
        
        if subfolder_summary_list:
            writer.writerow([])
            writer.writerow(["=== Subfolder Statistics Summary ==="])
            writer.writerow(["Subfolder Path", "Total Cells", "Category-1 Count", "Category-1 Percentage"])
            for sub_info in subfolder_summary_list:
                writer.writerow([
                    sub_info['path'], 
                    sub_info['total_all'], 
                    sub_info['cat1'],
                    f"{sub_info['cat1_pct']:.2f}%"
                ])

def create_distribution_plot(label_counts, output_path, title_prefix, cat1_count, total_all, cat1_pct, filter_info=None):
    """创建并保存标签分布柱状图"""
    plt.figure(figsize=(14, 7))
    counts = [label_counts.get(label, 0) for label in sorted_labels]
    bars = plt.bar(sorted_labels, counts, color='steelblue', edgecolor='black', linewidth=0.5)
    
    for i, label in enumerate(sorted_labels):
        big_cat = cell_dict_big.get(label)
        if big_cat == 0:
            bars[i].set_color('lightgray')
        elif big_cat == 1:
            bars[i].set_color('coral')
        elif big_cat == 2:
            bars[i].set_color('steelblue')
        else:
            bars[i].set_color('orange')
    
    plt.xlabel("Cell Type Label", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency (Valid Cells)", fontsize=12, fontweight='bold')
    
    subtitle = f"Category-1 Ratio: {cat1_pct:.2f}% ({cat1_count}/{total_all})"
    if filter_info:
        subtitle += f" | Filtered: Area≥{filter_info['area']}, Circ≥{filter_info['circ']}, CropSize≤{filter_info['crop_size']}"
        
    plt.title(f"Valid Label Distribution - {title_prefix}\n{subtitle}", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='coral', edgecolor='black', label='Category-1 (Target)'),
        Patch(facecolor='steelblue', edgecolor='black', label='Category-2 (Other)'),
        Patch(facecolor='lightgray', edgecolor='black', label='Category-0 (Background)'),
        Patch(facecolor='orange', edgecolor='black', label='Undefined Labels')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    for i, (label, count) in enumerate(zip(sorted_labels, counts)):
        if count > 0:
            plt.text(i, count, str(count), ha='center', va='bottom', 
                    fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_rejection_plot(reject_stats, total_raw, output_path, title_prefix, filter_info=None):
    """创建并保存剔除原因平行统计图"""
    reasons = ["area", "circularity", "edge", "crop_size"]
    counts = [reject_stats[r]["count"] for r in reasons]
    percentages = [reject_stats[r]["percentage"] for r in reasons]
    
    if sum(counts) == 0:
        return None
    
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a05195']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(reasons, counts, color=colors, edgecolor='black', linewidth=0.8)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel("Rejection Reason", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Rejected Cells", fontsize=12, fontweight='bold')
    
    subtitle = f"Total Raw: {total_raw} | Total Rejected: {sum(counts)}"
    if filter_info:
        subtitle += f" | Thresholds: Area≥{filter_info['area']}, Circ≥{filter_info['circ']}, Margin={filter_info['margin']}px, CropSize≤{filter_info['crop_size']}px"
    
    plt.title(f"Rejection Statistics - {title_prefix}\n{subtitle}", 
              fontsize=13, fontweight='bold', pad=15)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_path

def create_reject_by_label_heatmap(reject_by_label_stats, output_path, title_prefix, filter_info=None):
    """
    创建每个标签的剔除原因热力图（主目录级别）
    展示每个细胞类型因四种原因被筛除的比例
    """
    # 只选择有原始数据的标签
    labels_with_data = []
    area_data = []
    circ_data = []
    edge_data = []
    crop_data = []
    
    for label in sorted_labels:
        stats = reject_by_label_stats.get(label, {})
        raw_count = stats.get("raw_count", 0)
        if raw_count > 0:
            labels_with_data.append(label)
            area_data.append(stats.get("rejects", {}).get("area", {}).get("percentage", 0))
            circ_data.append(stats.get("rejects", {}).get("circularity", {}).get("percentage", 0))
            edge_data.append(stats.get("rejects", {}).get("edge", {}).get("percentage", 0))
            crop_data.append(stats.get("rejects", {}).get("crop_size", {}).get("percentage", 0))
    
    if len(labels_with_data) == 0:
        return None
    
    # 创建热力图数据
    data = np.array([area_data, circ_data, edge_data, crop_data])
    reasons = ["Area", "Circularity", "Edge", "CropSize"]
    
    plt.figure(figsize=(16, 8))
    im = plt.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # 设置坐标轴
    plt.yticks(range(4), reasons, fontsize=11, fontweight='bold')
    plt.xticks(range(len(labels_with_data)), labels_with_data, rotation=45, ha='right', fontsize=9)
    
    # 添加数值标注
    for i in range(4):
        for j in range(len(labels_with_data)):
            val = data[i, j]
            if val > 0:
                color = 'white' if val > 50 else 'black'
                plt.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color=color)
    
    # 添加颜色条
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label('Rejection Rate (%)', fontsize=11, fontweight='bold')
    
    # 标题
    subtitle = f"Thresholds: Area≥{filter_info['area'] if filter_info else MIN_AREA}, Circ≥{filter_info['circ'] if filter_info else MIN_CIRCULARITY}, Margin={filter_info['margin'] if filter_info else EDGE_MARGIN}px, CropSize≤{filter_info['crop_size'] if filter_info else CROP_SIZE}px"
    plt.title(f"Rejection Rate by Cell Type - {title_prefix}\n{subtitle}", 
              fontsize=13, fontweight='bold', pad=15)
    
    plt.xlabel("Cell Type Label", fontsize=12, fontweight='bold')
    plt.ylabel("Rejection Reason", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_path

# ================= 主处理流程 =================

# 缓存图像尺寸，避免重复读取
image_shape_cache = {}

for directory in directories:
    print(f"\n{'='*60}")
    print(f"正在处理目录：{directory}")
    print(f"{'='*60}")
    
    dir_label_counts = collections.defaultdict(int)
    dir_reject_counts = collections.defaultdict(int)
    dir_reject_counts_by_label = collections.defaultdict(lambda: collections.defaultdict(int))  # 【新增】按标签统计剔除原因
    dir_label_raw_counts = collections.defaultdict(int)  # 【新增】每个标签的原始数量
    dir_raw_count = 0
    dir_pic_counts = 0
    dir_subfolder_counts = 0
    subfolder_stats_collection = []
    
    for root, dirs, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        
        subfolder_label_counts = collections.defaultdict(int)
        subfolder_reject_counts = collections.defaultdict(int)
        subfolder_raw_count = 0
        subfolder_pic_counts = 0
        
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                subfolder_pic_counts += 1
                
                # 1. 寻找对应的图像文件
                img_path = find_matching_image(file_path, image_shape_cache)
                img_shape = get_image_shape(img_path, image_shape_cache)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue 
                        label = shape["label"]
                        points = shape.get("points")
                        subfolder_raw_count += 1
                        dir_label_raw_counts[label] += 1  # 【新增】记录原始标签计数
                        dir_raw_count += 1
                        
                        # 2. 应用筛选逻辑（获取剔除原因）
                        if img_shape and points:
                            is_valid, reject_reason = is_valid_cell(
                                points, 
                                img_shape, 
                                filter_edge=FILTER_EDGE_CELLS, 
                                margin=EDGE_MARGIN, 
                                min_area=MIN_AREA, 
                                min_circ=MIN_CIRCULARITY,
                                crop_size=CROP_SIZE
                            )
                            if is_valid:
                                subfolder_label_counts[label] += 1
                                dir_label_counts[label] += 1
                            else:
                                subfolder_reject_counts[reject_reason] += 1
                                dir_reject_counts[reject_reason] += 1
                                # 【新增】记录该标签因该原因被剔除
                                dir_reject_counts_by_label[label][reject_reason] += 1
                        else:
                            subfolder_reject_counts["invalid_input"] += 1
                            dir_reject_counts["invalid_input"] += 1
                            dir_reject_counts_by_label[label]["invalid_input"] += 1
                            
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")
                    subfolder_reject_counts["exception"] += 1
                    dir_reject_counts["exception"] += 1
                    dir_reject_counts_by_label[label]["exception"] += 1
        
        if subfolder_pic_counts > 0:
            dir_pic_counts += subfolder_pic_counts
            dir_subfolder_counts += 1
            
            # 计算子文件夹的类别统计
            s_cat0, s_cat1, s_cat2, s_undef, s_cat1_pct, s_total = calculate_category_stats(
                subfolder_label_counts, cell_dict_big
            )
            # 计算子文件夹的剔除统计
            s_reject_stats = calculate_reject_stats(subfolder_reject_counts, subfolder_raw_count)
            
            subfolder_stats_collection.append({
                'path': relative_path,
                'label_counts': dict(subfolder_label_counts),
                'reject_counts': dict(subfolder_reject_counts),
                'raw_count': subfolder_raw_count,
                'pic_counts': subfolder_pic_counts,
                'cat0': s_cat0,
                'cat1': s_cat1,
                'cat2': s_cat2,
                'undef': s_undef,
                'cat1_pct': s_cat1_pct,
                'total_all': s_total,
                'reject_stats': s_reject_stats
            })
    
    if dir_pic_counts > 0:
        # 计算主目录的类别统计
        cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all = calculate_category_stats(
            dir_label_counts, cell_dict_big
        )
        # 计算主目录的剔除统计
        reject_stats = calculate_reject_stats(dir_reject_counts, dir_raw_count)
        # 【新增】计算主目录按标签的剔除统计
        reject_by_label_stats = calculate_reject_by_label_stats(
            dir_reject_counts_by_label, dir_label_raw_counts
        )
        
        dataset_name = os.path.basename(directory)
        output_dir = os.path.join(GLOBAL_OUTPUT_ROOT, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        filter_info = {
            'area': MIN_AREA,
            'circ': MIN_CIRCULARITY,
            'margin': EDGE_MARGIN,
            'crop_size': CROP_SIZE
        }
        
        print(f"\n📊 目录统计摘要 (已过滤):")
        print(f"   子文件夹数：{dir_subfolder_counts}")
        print(f"   图片总数：{dir_pic_counts}")
        print(f"   原始标注总数：{dir_raw_count}")
        print(f"   有效细胞数：{total_all}")
        print(f"   剔除细胞数：{dir_raw_count - total_all}")
        print(f"   ├─ 面积过小：{reject_stats['area']['count']} ({reject_stats['area']['percentage']:.2f}%)")
        print(f"   ├─ 圆度不足：{reject_stats['circularity']['count']} ({reject_stats['circularity']['percentage']:.2f}%)")
        print(f"   ├─ 边缘细胞：{reject_stats['edge']['count']} ({reject_stats['edge']['percentage']:.2f}%)")
        print(f"   ├─ 超过裁剪尺寸：{reject_stats['crop_size']['count']} ({reject_stats['crop_size']['percentage']:.2f}%)")
        print(f"   🎯 类别 1 细胞占比：{cat1_percentage:.2f}%")
        
        # 1. 有效标签分布图
        output_filename_png = f"{dataset_name}_valid_distribution.png"
        output_path_png = os.path.join(output_dir, output_filename_png)
        create_distribution_plot(
            dir_label_counts, output_path_png, os.path.basename(directory),
            cat1_count, total_all, cat1_percentage, filter_info=filter_info
        )
        print(f"\n✅ 有效分布图已保存至：{output_path_png}")
        
        # 2. 剔除原因平行统计图
        reject_png_name = f"{dataset_name}_rejection_stats.png"
        reject_png_path = os.path.join(output_dir, reject_png_name)
        reject_result = create_rejection_plot(
            reject_stats, dir_raw_count, reject_png_path, 
            os.path.basename(directory), filter_info=filter_info
        )
        if reject_result:
            print(f"✅ 剔除统计图已保存至：{reject_png_path}")
        
        # 3. 【新增】主目录 - 每个标签的剔除原因热力图
        heatmap_png_name = f"{dataset_name}_reject_by_label_heatmap.png"
        heatmap_png_path = os.path.join(output_dir, heatmap_png_name)
        heatmap_result = create_reject_by_label_heatmap(
            reject_by_label_stats, heatmap_png_path,
            os.path.basename(directory), filter_info=filter_info
        )
        if heatmap_result:
            print(f"✅ 标签剔除热力图已保存至：{heatmap_png_path}")
        else:
            print(f"⚠️  无标签剔除数据，跳过生成热力图")
        
        # 4. 主目录 CSV（含各标签剔除明细）
        output_filename_csv = f"{dataset_name}_valid_statistics.csv"
        output_path_csv = os.path.join(output_dir, output_filename_csv)
        write_detailed_csv(
            output_path_csv, os.path.basename(directory), dir_label_counts,
            dir_pic_counts, dir_subfolder_counts,
            cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all,
            reject_stats, dir_raw_count,
            reject_by_label_stats, dir_label_raw_counts,  # 【新增】
            subfolder_summary_list=subfolder_stats_collection,
            filter_info=filter_info
        )
        print(f"✅ 主目录 CSV 统计文件已保存至：{output_path_csv}")
    
        
        # 5. 子文件夹独立统计（不生成热力图，只生成主目录的）
        print(f"\n   正在生成 {len(subfolder_stats_collection)} 个子文件夹的独立统计...")
        for idx, sub_info in enumerate(subfolder_stats_collection, 1):
            safe_sub_name = sub_info['path'].replace(os.sep, '_').replace('.', 'root')
            
            # 5.1 子文件夹 CSV
            sub_csv_name = f"subfolder_{safe_sub_name}_stats.csv"
            sub_csv_path = os.path.join(output_dir, sub_csv_name)
            write_detailed_csv(
                sub_csv_path,
                f"{os.path.basename(directory)} / {sub_info['path']}",
                sub_info['label_counts'],
                sub_info['pic_counts'], 1,
                sub_info['cat0'], sub_info['cat1'], sub_info['cat2'], 
                sub_info['undef'], sub_info['cat1_pct'], sub_info['total_all'],
                sub_info['reject_stats'], sub_info['raw_count'],
                {}, {},  # 子文件夹不生成按标签剔除统计
                subfolder_summary_list=None, filter_info=filter_info
            )
            
            # 5.2 子文件夹 - 有效分布图
            sub_png_name = f"subfolder_{safe_sub_name}_distribution.png"
            sub_png_path = os.path.join(output_dir, sub_png_name)
            create_distribution_plot(
                sub_info['label_counts'], sub_png_path,
                f"{os.path.basename(directory)} / {sub_info['path']}",
                sub_info['cat1'], sub_info['total_all'],
                sub_info['cat1_pct'], filter_info=filter_info
            )
            
            # 5.3 子文件夹 - 剔除原因图
            sub_reject_png = f"subfolder_{safe_sub_name}_rejection.png"
            sub_reject_path = os.path.join(output_dir, sub_reject_png)
            create_rejection_plot(
                sub_info['reject_stats'], sub_info['raw_count'],
                sub_reject_path, f"{os.path.basename(directory)} / {sub_info['path']}",
                filter_info=filter_info
            )
            
            print(f"   [{idx}/{len(subfolder_stats_collection)}] {sub_info['path']} - CSV & PNGs 已生成")
        
        print(f"   ✅ 所有子文件夹统计生成完成。")

    else:
        print(f"\n⚠ 警告：该目录下未找到任何有效标注数据")

# ============ 全局汇总 ============
print(f"\n{'='*60}")
print("正在生成所有目录的汇总统计 CSV...")
print(f"{'='*60}")

all_datasets_stats = []
total_images_all = 0
total_raw_all = 0
total_cat0_all = 0
total_cat1_all = 0
total_cat2_all = 0
total_undefined_all = 0
total_reject_area_all = 0
total_reject_circ_all = 0
total_reject_edge_all = 0
total_reject_crop_all = 0

for directory in directories:
    dir_label_counts = collections.defaultdict(int)
    dir_reject_counts = collections.defaultdict(int)
    dir_reject_counts_by_label = collections.defaultdict(lambda: collections.defaultdict(int))
    dir_label_raw_counts = collections.defaultdict(int)
    dir_raw_count = 0
    dir_pic_counts = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                dir_pic_counts += 1
                
                img_path = find_matching_image(file_path, image_shape_cache)
                img_shape = get_image_shape(img_path, image_shape_cache)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue
                        label = shape["label"]
                        points = shape.get("points")
                        dir_raw_count += 1
                        dir_label_raw_counts[label] += 1
                        
                        if img_shape and points:
                            is_valid, reject_reason = is_valid_cell(
                                points, img_shape, FILTER_EDGE_CELLS, EDGE_MARGIN, MIN_AREA, MIN_CIRCULARITY, CROP_SIZE
                            )
                            if is_valid:
                                dir_label_counts[label] += 1
                            else:
                                dir_reject_counts[reject_reason] += 1
                                dir_reject_counts_by_label[label][reject_reason] += 1
                        else:
                            dir_reject_counts["invalid_input"] += 1
                            dir_reject_counts_by_label[label]["invalid_input"] += 1
                except Exception as e:
                    dir_reject_counts["exception"] += 1
    
    cat0, cat1, cat2, undefined, cat1_pct, total_all = calculate_category_stats(dir_label_counts, cell_dict_big)
    reject_stats = calculate_reject_stats(dir_reject_counts, dir_raw_count)
    
    all_datasets_stats.append({
        'dataset': os.path.basename(directory),
        'images': dir_pic_counts,
        'raw_annotations': dir_raw_count,
        'valid_cells': total_all,
        'cat0_count': cat0,
        'cat1_count': cat1,
        'cat2_count': cat2,
        'undefined_count': undefined,
        'total_all': total_all,
        'cat1_percentage': cat1_pct,
        'reject_area': reject_stats['area']['count'],
        'reject_circ': reject_stats['circularity']['count'],
        'reject_edge': reject_stats['edge']['count'],
        'reject_crop': reject_stats['crop_size']['count']
    })
    total_images_all += dir_pic_counts
    total_raw_all += dir_raw_count
    total_cat0_all += cat0
    total_cat1_all += cat1
    total_cat2_all += cat2
    total_undefined_all += undefined
    total_reject_area_all += reject_stats['area']['count']
    total_reject_circ_all += reject_stats['circularity']['count']
    total_reject_edge_all += reject_stats['edge']['count']
    total_reject_crop_all += reject_stats['crop_size']['count']

global_output_dir = os.path.dirname(directories[0])
if not os.path.exists(global_output_dir):
    global_output_dir = "./global_statistics"
    os.makedirs(global_output_dir, exist_ok=True)

global_csv_path = os.path.join(GLOBAL_OUTPUT_ROOT, "ALL_DATASETS_VALID_SUMMARY.csv")
with open(global_csv_path, "w", encoding="utf-8-sig", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["=== Global Summary (Valid Cells Only) ==="])
    writer.writerow(["Filter Settings", f"Area>={MIN_AREA}, Circ>={MIN_CIRCULARITY}, Margin={EDGE_MARGIN}, CropSize<={CROP_SIZE}"])
    writer.writerow(["Total Datasets", len(directories)])
    writer.writerow(["Total Images", total_images_all])
    writer.writerow(["Total Raw Annotations", total_raw_all])
    writer.writerow(["Total Valid Cells", total_cat0_all + total_cat1_all + total_cat2_all + total_undefined_all])
    writer.writerow(["Total Rejected Cells", total_raw_all - (total_cat0_all + total_cat1_all + total_cat2_all + total_undefined_all)])
    writer.writerow([])
    writer.writerow(["=== Rejection Reasons (Global) ==="])
    writer.writerow(["Reason", "Count", "Percentage of Raw"])
    writer.writerow(["Area < threshold", total_reject_area_all, f"{(total_reject_area_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
    writer.writerow(["Circularity < threshold", total_reject_circ_all, f"{(total_reject_circ_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
    writer.writerow(["Edge margin", total_reject_edge_all, f"{(total_reject_edge_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
    writer.writerow(["Crop size exceed", total_reject_crop_all, f"{(total_reject_crop_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
    writer.writerow([])
    writer.writerow(["Dataset", "Images", "Raw Annotations", "Valid Cells", "Rejected", "Cat-1 Count", "Cat-1 %", 
                    "Reject(Area)", "Reject(Circ)", "Reject(Edge)", "Reject(CropSize)"])
    for ds in all_datasets_stats:
        rejected = ds['raw_annotations'] - ds['valid_cells']
        writer.writerow([
            ds['dataset'], ds['images'], ds['raw_annotations'], ds['valid_cells'], rejected,
            ds['cat1_count'], f"{ds['cat1_percentage']:.2f}%",
            ds['reject_area'], ds['reject_circ'], ds['reject_edge'], ds['reject_crop']
        ])

print(f"\n{'='*60}")
print("✅ 所有目录处理完成！")
print(f"✅ 全局汇总 CSV 已保存至：{global_csv_path}")
print(f"{'='*60}")



# import os
# import json
# import collections
# import matplotlib.pyplot as plt
# import csv
# import cv2
# import numpy as np
# import math

# '''
# 统计每个指定文件夹的总标签分布（汇总该文件夹下所有子文件夹）
# 修改版：加入细胞筛选逻辑（边缘、面积、圆度），统计有效细胞分布
# 细化版：平行统计三种筛选原因的剔除情况
# 增强版：统计每个细胞类型因三种原因被筛除的数目（主目录级别）
# '''

# # ================= 配置区域 =================
# # 筛选阈值 (与裁剪脚本保持一致)
# MIN_AREA = 10000          # 最小面积
# MIN_CIRCULARITY = 0.6     # 最小圆度
# EDGE_MARGIN = 10          # 边缘容差像素
# FILTER_EDGE_CELLS = True  # 是否启用边缘细胞过滤

# # 指定要遍历的目录
# directories = [
#     "/root/autodl-tmp/data/FXH_imgs_260211",
#     "/root/autodl-tmp/data/BJH_imgs_260211",
#     "/root/autodl-tmp/data/BEPH_imgs_260211",
#     "/root/autodl-tmp/data/MAIN_imgs_outline_mask_260211",
#     "/root/autodl-tmp/data/MAIN_imgs_dotnoutline_mask_260211",
# ]

# GLOBAL_OUTPUT_ROOT = "/root/autodl-tmp/data/statistics_plots_filtered_0.6"
# os.makedirs(GLOBAL_OUTPUT_ROOT, exist_ok=True)

# # 原始标签顺序定义
# cell_dict = {
#     "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
#     "E": 8, "B": 9, "M0": 10, "M": 11, "M1": 12, "M2": 13,
#     "R": 14, "R1": 15, "R2": 16, "R3": 17,
#     "J": 18, "J1": 19, "J2": 20, "J3": 21, "J4": 22,
#     "L": 23, "L1": 24, "L2": 25, "L3": 26, "L4": 27,
#     "P": 28, "P1": 29, "P2": 30, "P3": 31,
#     "B1": 32, "E1": 33, "A": 34, "F": 35, "V": 36, "0": 36
# }

# # 按照特定顺序排序的标签列表
# sorted_labels = sorted(cell_dict.keys(), key=lambda x: cell_dict[x])

# # 大类映射字典
# cell_dict_big = {
#     "V": 0, "0": 0,
#     "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
#     "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
#     "E": 2, "B": 2, "E1": 2, "B1": 2,
#     "M0": 2, "M2": 2, "R2": 2, "R3": 2,
#     "J2": 2, "J3": 2, "J4": 2,
#     "P": 2, "P1": 2, "P2": 2, "P3": 2,
#     "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
# }

# # ================= 辅助函数 =================

# def calculate_circularity(contour):
#     """计算轮廓的圆度"""
#     try:
#         if contour.dtype != np.int32:
#             contour = contour.astype(np.int32)
#         area = cv2.contourArea(contour)
#         if area == 0:
#             return 0
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter == 0:
#             return 0
#         return (4 * math.pi * area) / (perimeter * perimeter)
#     except Exception:
#         return 0

# def get_image_shape(image_path, cache):
#     """获取图像尺寸，带缓存"""
#     if image_path in cache:
#         return cache[image_path]
    
#     if not os.path.exists(image_path):
#         return None
    
#     # 屏蔽 OpenCV JPEG 警告
#     import os as _os
#     _os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
    
#     h, w = img.shape[:2]
#     cache[image_path] = (h, w)
#     return (h, w)

# def find_matching_image(json_path, image_dir_cache):
#     """根据 JSON 路径寻找对应的图像文件"""
#     base_name = os.path.splitext(os.path.basename(json_path))[0]
#     parent_dir = os.path.dirname(json_path)
    
#     for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG']:
#         potential_path = os.path.join(parent_dir, base_name + ext)
#         if os.path.exists(potential_path):
#             return potential_path
#     return None

# def is_valid_cell(points, image_shape, filter_edge=True, margin=10, min_area=10000, min_circ=0.7):
#     """
#     检查细胞是否满足所有筛选条件
#     Returns:
#         (is_valid: bool, reject_reason: str or None)
#         reject_reason: None | "area" | "circularity" | "edge"
#     """
#     if not points or image_shape is None:
#         return False, "invalid_input"
    
#     try:
#         # 【关键】：强制 int32 + 内存连续，兼容新架构
#         contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
#         contour = np.ascontiguousarray(contour)

#         # 1. 面积检查
#         area = cv2.contourArea(contour)
#         if area < min_area:
#             return False, "area"
        
#         # 2. 圆度检查
#         circ = calculate_circularity(contour)
#         if circ < min_circ:
#             return False, "circularity"
        
#         # 3. 边缘检查
#         if filter_edge:
#             h, w = image_shape
#             x, y, bw, bh = cv2.boundingRect(contour)
#             if (x <= margin or 
#                 x + bw >= w - margin or 
#                 y <= margin or 
#                 y + bh >= h - margin):
#                 return False, "edge"
                
#         return True, None
#     except Exception as e:
#         return False, "exception"

# def calculate_category_stats(label_counts, cell_dict_big):
#     """根据 cell_dict_big 映射计算各类别统计"""
#     cat0_count = 0
#     cat1_count = 0
#     cat2_count = 0
#     undefined_count = 0
    
#     for label, count in label_counts.items():
#         category = cell_dict_big.get(label, -1)
#         if category == 0:
#             cat0_count += count
#         elif category == 1:
#             cat1_count += count
#         elif category == 2:
#             cat2_count += count
#         else:
#             undefined_count += count
    
#     total_all = cat0_count + cat1_count + cat2_count + undefined_count
#     percentage = (cat1_count / total_all * 100) if total_all > 0 else 0.0
    
#     return cat0_count, cat1_count, cat2_count, undefined_count, percentage, total_all

# def calculate_reject_stats(reject_counts, total_raw):
#     """计算剔除统计的百分比"""
#     stats = {}
#     for reason in ["area", "circularity", "edge", "invalid_input", "exception"]:
#         count = reject_counts.get(reason, 0)
#         pct = (count / total_raw * 100) if total_raw > 0 else 0
#         stats[reason] = {"count": count, "percentage": pct}
#     return stats

# def calculate_reject_by_label_stats(reject_counts_by_label, label_raw_counts):
#     """
#     计算每个标签的剔除原因统计
#     Returns:
#         dict: {label: {reason: {"count": int, "percentage": float}}}
#     """
#     stats = {}
#     for label in sorted_labels:
#         raw_count = label_raw_counts.get(label, 0)
#         label_rejects = reject_counts_by_label.get(label, {})
        
#         stats[label] = {
#             "raw_count": raw_count,
#             "rejects": {}
#         }
        
#         for reason in ["area", "circularity", "edge"]:
#             count = label_rejects.get(reason, 0)
#             pct = (count / raw_count * 100) if raw_count > 0 else 0
#             stats[label]["rejects"][reason] = {
#                 "count": count,
#                 "percentage": pct
#             }
    
#     return stats

# def write_detailed_csv(output_path, dataset_name, label_counts, dir_pic_counts, dir_subfolder_counts, 
#                        cat0, cat1, cat2, undefined_count, cat1_pct, total_all, 
#                        reject_stats, total_raw_annotations,
#                        reject_by_label_stats, label_raw_counts,
#                        subfolder_summary_list=None, filter_info=None):
#     """写入详细的 CSV 统计文件（含剔除原因统计）"""
#     with open(output_path, "w", encoding="utf-8-sig", newline='') as csvfile:
#         writer = csv.writer(csvfile)
        
#         # 1. 文件头信息
#         writer.writerow(["Dataset", dataset_name])
#         if filter_info:
#             writer.writerow(["Filter Settings", f"Area>={filter_info['area']}, Circ>={filter_info['circ']}, Margin={filter_info['margin']}"])
#         writer.writerow(["Total Images", dir_pic_counts])
#         writer.writerow(["Total Subfolders", dir_subfolder_counts])
#         writer.writerow(["Total Raw Annotations", total_raw_annotations])
#         writer.writerow(["Total Valid Cells", total_all])
#         writer.writerow(["Total Rejected Cells", total_raw_annotations - total_all])
#         writer.writerow([])
        
#         # 2. 剔除原因平行统计
#         writer.writerow(["=== Rejection Reasons (Parallel Statistics) ==="])
#         writer.writerow(["Reason", "Description", "Count", "Percentage of Raw"])
#         reason_desc = {
#             "area": f"Area < {filter_info['area'] if filter_info else MIN_AREA}",
#             "circularity": f"Circularity < {filter_info['circ'] if filter_info else MIN_CIRCULARITY}",
#             "edge": f"Within {filter_info['margin'] if filter_info else EDGE_MARGIN}px of image border",
#             "invalid_input": "Invalid points or missing image",
#             "exception": "Processing exception"
#         }
#         for reason in ["area", "circularity", "edge", "invalid_input", "exception"]:
#             if reason in reject_stats:
#                 s = reject_stats[reason]
#                 writer.writerow([reason, reason_desc.get(reason, ""), s["count"], f"{s['percentage']:.2f}%"])
#         writer.writerow([])
        
#         # 3. 【新增】各标签剔除原因明细表
#         writer.writerow(["=== Rejection by Cell Type (Label-Specific) ==="])
#         writer.writerow(["Label", "Raw Count", "Valid Count", "Rejected", 
#                         "Reject(Area)", "Reject(Circ)", "Reject(Edge)", 
#                         "Area%", "Circ%", "Edge%"])
        
#         for label in sorted_labels:
#             raw_count = label_raw_counts.get(label, 0)
#             valid_count = label_counts.get(label, 0)
#             rejected = raw_count - valid_count
            
#             if raw_count > 0:
#                 label_stats = reject_by_label_stats.get(label, {})
#                 area_reject = label_stats.get("rejects", {}).get("area", {}).get("count", 0)
#                 circ_reject = label_stats.get("rejects", {}).get("circularity", {}).get("count", 0)
#                 edge_reject = label_stats.get("rejects", {}).get("edge", {}).get("count", 0)
                
#                 area_pct = label_stats.get("rejects", {}).get("area", {}).get("percentage", 0)
#                 circ_pct = label_stats.get("rejects", {}).get("circularity", {}).get("percentage", 0)
#                 edge_pct = label_stats.get("rejects", {}).get("edge", {}).get("percentage", 0)
                
#                 writer.writerow([
#                     label, raw_count, valid_count, rejected,
#                     area_reject, circ_reject, edge_reject,
#                     f"{area_pct:.2f}%", f"{circ_pct:.2f}%", f"{edge_pct:.2f}%"
#                 ])
#         writer.writerow([])
        
#         # 4. 类别统计汇总
#         writer.writerow(["=== Category Statistics (by cell_dict_big) ==="])
#         writer.writerow(["Category", "Description", "Count", "Percentage of Valid Cells"])
        
#         total_for_pct = total_all if total_all > 0 else 1
#         writer.writerow(["0", "Background (V, 0)", cat0, f"{cat0/total_for_pct*100:.2f}%"])
#         writer.writerow(["1", "Target Cells (N, N1, M, M1, R, R1, J, J1)", cat1, f"{cat1_pct:.2f}%"])
#         writer.writerow(["2", "Other Cells", cat2, f"{cat2/total_for_pct*100:.2f}%"])
        
#         if undefined_count > 0:
#             writer.writerow(["Undefined", "Labels not in cell_dict_big (Total)", undefined_count, f"{undefined_count/total_for_pct*100:.2f}%"])
#             writer.writerow(["--- Undefined Label Details ---", "", "", ""])
#             undefined_labels = {k: v for k, v in label_counts.items() if k not in cell_dict_big}
#             sorted_undefined = sorted(undefined_labels.keys())
#             for u_label in sorted_undefined:
#                 u_count = undefined_labels[u_label]
#                 u_pct = (u_count / total_all * 100) if total_all > 0 else 0
#                 writer.writerow(["", f"  -> {u_label}", u_count, f"{u_pct:.2f}%"])
#         writer.writerow([])
        
#         # 5. 各标签详细统计 (已知标签)
#         writer.writerow(["=== Label Details (Defined) ==="])
#         writer.writerow(["Label", "Order", "BigCategory", "Count", "Percentage of Valid"])
        
#         for label in sorted_labels:
#             count = label_counts.get(label, 0)
#             if count > 0:
#                 pct_total = (count / total_all * 100) if total_all > 0 else 0
#                 big_cat = cell_dict_big.get(label, "N/A")
#                 writer.writerow([
#                     label, 
#                     cell_dict[label], 
#                     big_cat,
#                     count, 
#                     f"{pct_total:.2f}%"
#                 ])
        
#         if subfolder_summary_list:
#             writer.writerow([])
#             writer.writerow(["=== Subfolder Statistics Summary ==="])
#             writer.writerow(["Subfolder Path", "Total Cells", "Category-1 Count", "Category-1 Percentage"])
#             for sub_info in subfolder_summary_list:
#                 writer.writerow([
#                     sub_info['path'], 
#                     sub_info['total_all'], 
#                     sub_info['cat1'],
#                     f"{sub_info['cat1_pct']:.2f}%"
#                 ])

# def create_distribution_plot(label_counts, output_path, title_prefix, cat1_count, total_all, cat1_pct, filter_info=None):
#     """创建并保存标签分布柱状图"""
#     plt.figure(figsize=(14, 7))
#     counts = [label_counts.get(label, 0) for label in sorted_labels]
#     bars = plt.bar(sorted_labels, counts, color='steelblue', edgecolor='black', linewidth=0.5)
    
#     for i, label in enumerate(sorted_labels):
#         big_cat = cell_dict_big.get(label)
#         if big_cat == 0:
#             bars[i].set_color('lightgray')
#         elif big_cat == 1:
#             bars[i].set_color('coral')
#         elif big_cat == 2:
#             bars[i].set_color('steelblue')
#         else:
#             bars[i].set_color('orange')
    
#     plt.xlabel("Cell Type Label", fontsize=12, fontweight='bold')
#     plt.ylabel("Frequency (Valid Cells)", fontsize=12, fontweight='bold')
    
#     subtitle = f"Category-1 Ratio: {cat1_pct:.2f}% ({cat1_count}/{total_all})"
#     if filter_info:
#         subtitle += f" | Filtered: Area≥{filter_info['area']}, Circ≥{filter_info['circ']}"
        
#     plt.title(f"Valid Label Distribution - {title_prefix}\n{subtitle}", 
#               fontsize=13, fontweight='bold', pad=15)
#     plt.xticks(rotation=45, ha='right')
#     plt.grid(axis='y', linestyle='--', alpha=0.3)
    
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='coral', edgecolor='black', label='Category-1 (Target)'),
#         Patch(facecolor='steelblue', edgecolor='black', label='Category-2 (Other)'),
#         Patch(facecolor='lightgray', edgecolor='black', label='Category-0 (Background)'),
#         Patch(facecolor='orange', edgecolor='black', label='Undefined Labels')
#     ]
#     plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
#     for i, (label, count) in enumerate(zip(sorted_labels, counts)):
#         if count > 0:
#             plt.text(i, count, str(count), ha='center', va='bottom', 
#                     fontsize=7, fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     plt.close()

# def create_rejection_plot(reject_stats, total_raw, output_path, title_prefix, filter_info=None):
#     """创建并保存剔除原因平行统计图"""
#     reasons = ["area", "circularity", "edge"]
#     counts = [reject_stats[r]["count"] for r in reasons]
#     percentages = [reject_stats[r]["percentage"] for r in reasons]
    
#     if sum(counts) == 0:
#         return None
    
#     colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
    
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(reasons, counts, color=colors, edgecolor='black', linewidth=0.8)
    
#     for bar, count, pct in zip(bars, counts, percentages):
#         height = bar.get_height()
#         if height > 0:
#             plt.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{count}\n({pct:.1f}%)',
#                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     plt.xlabel("Rejection Reason", fontsize=12, fontweight='bold')
#     plt.ylabel("Number of Rejected Cells", fontsize=12, fontweight='bold')
    
#     subtitle = f"Total Raw: {total_raw} | Total Rejected: {sum(counts)}"
#     if filter_info:
#         subtitle += f" | Thresholds: Area≥{filter_info['area']}, Circ≥{filter_info['circ']}, Margin={filter_info['margin']}px"
    
#     plt.title(f"Rejection Statistics - {title_prefix}\n{subtitle}", 
#               fontsize=13, fontweight='bold', pad=15)
    
#     plt.grid(axis='y', linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     plt.close()
#     return output_path

# def create_reject_by_label_heatmap(reject_by_label_stats, output_path, title_prefix, filter_info=None):
#     """
#     创建每个标签的剔除原因热力图（主目录级别）
#     展示每个细胞类型因三种原因被筛除的比例
#     """
#     # 只选择有原始数据的标签
#     labels_with_data = []
#     area_data = []
#     circ_data = []
#     edge_data = []
    
#     for label in sorted_labels:
#         stats = reject_by_label_stats.get(label, {})
#         raw_count = stats.get("raw_count", 0)
#         if raw_count > 0:
#             labels_with_data.append(label)
#             area_data.append(stats.get("rejects", {}).get("area", {}).get("percentage", 0))
#             circ_data.append(stats.get("rejects", {}).get("circularity", {}).get("percentage", 0))
#             edge_data.append(stats.get("rejects", {}).get("edge", {}).get("percentage", 0))
    
#     if len(labels_with_data) == 0:
#         return None
    
#     # 创建热力图数据
#     data = np.array([area_data, circ_data, edge_data])
#     reasons = ["Area", "Circularity", "Edge"]
    
#     plt.figure(figsize=(14, 8))
#     im = plt.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
#     # 设置坐标轴
#     plt.yticks(range(3), reasons, fontsize=11, fontweight='bold')
#     plt.xticks(range(len(labels_with_data)), labels_with_data, rotation=45, ha='right', fontsize=9)
    
#     # 添加数值标注
#     for i in range(3):
#         for j in range(len(labels_with_data)):
#             val = data[i, j]
#             if val > 0:
#                 color = 'white' if val > 50 else 'black'
#                 plt.text(j, i, f'{val:.1f}%', ha='center', va='center', 
#                         fontsize=8, fontweight='bold', color=color)
    
#     # 添加颜色条
#     cbar = plt.colorbar(im, pad=0.02)
#     cbar.set_label('Rejection Rate (%)', fontsize=11, fontweight='bold')
    
#     # 标题
#     subtitle = f"Thresholds: Area≥{filter_info['area'] if filter_info else MIN_AREA}, Circ≥{filter_info['circ'] if filter_info else MIN_CIRCULARITY}, Margin={filter_info['margin'] if filter_info else EDGE_MARGIN}px"
#     plt.title(f"Rejection Rate by Cell Type - {title_prefix}\n{subtitle}", 
#               fontsize=13, fontweight='bold', pad=15)
    
#     plt.xlabel("Cell Type Label", fontsize=12, fontweight='bold')
#     plt.ylabel("Rejection Reason", fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     plt.close()
#     return output_path

# # ================= 主处理流程 =================

# # 缓存图像尺寸，避免重复读取
# image_shape_cache = {}

# for directory in directories:
#     print(f"\n{'='*60}")
#     print(f"正在处理目录：{directory}")
#     print(f"{'='*60}")
    
#     dir_label_counts = collections.defaultdict(int)
#     dir_reject_counts = collections.defaultdict(int)
#     dir_reject_counts_by_label = collections.defaultdict(lambda: collections.defaultdict(int))  # 【新增】按标签统计剔除原因
#     dir_label_raw_counts = collections.defaultdict(int)  # 【新增】每个标签的原始数量
#     dir_raw_count = 0
#     dir_pic_counts = 0
#     dir_subfolder_counts = 0
#     subfolder_stats_collection = []
    
#     for root, dirs, files in os.walk(directory):
#         relative_path = os.path.relpath(root, directory)
        
#         subfolder_label_counts = collections.defaultdict(int)
#         subfolder_reject_counts = collections.defaultdict(int)
#         subfolder_raw_count = 0
#         subfolder_pic_counts = 0
        
#         for filename in files:
#             if filename.endswith(".json"):
#                 file_path = os.path.join(root, filename)
#                 subfolder_pic_counts += 1
                
#                 # 1. 寻找对应的图像文件
#                 img_path = find_matching_image(file_path, image_shape_cache)
#                 img_shape = get_image_shape(img_path, image_shape_cache)
                
#                 try:
#                     with open(file_path, "r", encoding="utf-8") as f:
#                         data = json.load(f)
                    
#                     for shape in data["shapes"]:
#                         if shape.get("shape_type") != "polygon":
#                             continue 
#                         label = shape["label"]
#                         points = shape.get("points")
#                         subfolder_raw_count += 1
#                         dir_label_raw_counts[label] += 1  # 【新增】记录原始标签计数
                        
#                         # 2. 应用筛选逻辑（获取剔除原因）
#                         if img_shape and points:
#                             is_valid, reject_reason = is_valid_cell(
#                                 points, 
#                                 img_shape, 
#                                 filter_edge=FILTER_EDGE_CELLS, 
#                                 margin=EDGE_MARGIN, 
#                                 min_area=MIN_AREA, 
#                                 min_circ=MIN_CIRCULARITY
#                             )
#                             if is_valid:
#                                 subfolder_label_counts[label] += 1
#                                 dir_label_counts[label] += 1
#                             else:
#                                 subfolder_reject_counts[reject_reason] += 1
#                                 dir_reject_counts[reject_reason] += 1
#                                 # 【新增】记录该标签因该原因被剔除
#                                 dir_reject_counts_by_label[label][reject_reason] += 1
#                         else:
#                             subfolder_reject_counts["invalid_input"] += 1
#                             dir_reject_counts["invalid_input"] += 1
#                             dir_reject_counts_by_label[label]["invalid_input"] += 1
                            
#                 except Exception as e:
#                     print(f"处理文件 {file_path} 时出错：{e}")
#                     subfolder_reject_counts["exception"] += 1
#                     dir_reject_counts["exception"] += 1
#                     dir_reject_counts_by_label[label]["exception"] += 1
        
#         if subfolder_pic_counts > 0:
#             dir_pic_counts += subfolder_pic_counts
#             dir_subfolder_counts += 1
            
#             # 计算子文件夹的类别统计
#             s_cat0, s_cat1, s_cat2, s_undef, s_cat1_pct, s_total = calculate_category_stats(
#                 subfolder_label_counts, cell_dict_big
#             )
#             # 计算子文件夹的剔除统计
#             s_reject_stats = calculate_reject_stats(subfolder_reject_counts, subfolder_raw_count)
            
#             subfolder_stats_collection.append({
#                 'path': relative_path,
#                 'label_counts': dict(subfolder_label_counts),
#                 'reject_counts': dict(subfolder_reject_counts),
#                 'raw_count': subfolder_raw_count,
#                 'pic_counts': subfolder_pic_counts,
#                 'cat0': s_cat0,
#                 'cat1': s_cat1,
#                 'cat2': s_cat2,
#                 'undef': s_undef,
#                 'cat1_pct': s_cat1_pct,
#                 'total_all': s_total,
#                 'reject_stats': s_reject_stats
#             })
    
#     if dir_pic_counts > 0:
#         # 计算主目录的类别统计
#         cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all = calculate_category_stats(
#             dir_label_counts, cell_dict_big
#         )
#         # 计算主目录的剔除统计
#         reject_stats = calculate_reject_stats(dir_reject_counts, dir_raw_count)
#         # 【新增】计算主目录按标签的剔除统计
#         reject_by_label_stats = calculate_reject_by_label_stats(
#             dir_reject_counts_by_label, dir_label_raw_counts
#         )
        
#         dataset_name = os.path.basename(directory)
#         output_dir = os.path.join(GLOBAL_OUTPUT_ROOT, dataset_name)
#         os.makedirs(output_dir, exist_ok=True)

#         filter_info = {
#             'area': MIN_AREA,
#             'circ': MIN_CIRCULARITY,
#             'margin': EDGE_MARGIN
#         }
        
#         print(f"\n📊 目录统计摘要 (已过滤):")
#         print(f"   子文件夹数：{dir_subfolder_counts}")
#         print(f"   图片总数：{dir_pic_counts}")
#         print(f"   原始标注总数：{dir_raw_count}")
#         print(f"   有效细胞数：{total_all}")
#         print(f"   剔除细胞数：{dir_raw_count - total_all}")
#         print(f"   ├─ 面积过小：{reject_stats['area']['count']} ({reject_stats['area']['percentage']:.2f}%)")
#         print(f"   ├─ 圆度不足：{reject_stats['circularity']['count']} ({reject_stats['circularity']['percentage']:.2f}%)")
#         print(f"   ├─ 边缘细胞：{reject_stats['edge']['count']} ({reject_stats['edge']['percentage']:.2f}%)")
#         print(f"   🎯 类别 1 细胞占比：{cat1_percentage:.2f}%")
        
#         # 1. 有效标签分布图
#         output_filename_png = f"{dataset_name}_valid_distribution.png"
#         output_path_png = os.path.join(output_dir, output_filename_png)
#         create_distribution_plot(
#             dir_label_counts, output_path_png, os.path.basename(directory),
#             cat1_count, total_all, cat1_percentage, filter_info=filter_info
#         )
#         print(f"\n✅ 有效分布图已保存至：{output_path_png}")
        
#         # 2. 剔除原因平行统计图
#         reject_png_name = f"{dataset_name}_rejection_stats.png"
#         reject_png_path = os.path.join(output_dir, reject_png_name)
#         reject_result = create_rejection_plot(
#             reject_stats, dir_raw_count, reject_png_path, 
#             os.path.basename(directory), filter_info=filter_info
#         )
#         if reject_result:
#             print(f"✅ 剔除统计图已保存至：{reject_png_path}")
        
#         # 3. 【新增】主目录 - 每个标签的剔除原因热力图
#         heatmap_png_name = f"{dataset_name}_reject_by_label_heatmap.png"
#         heatmap_png_path = os.path.join(output_dir, heatmap_png_name)
#         heatmap_result = create_reject_by_label_heatmap(
#             reject_by_label_stats, heatmap_png_path,
#             os.path.basename(directory), filter_info=filter_info
#         )
#         if heatmap_result:
#             print(f"✅ 标签剔除热力图已保存至：{heatmap_png_path}")
#         else:
#             print(f"⚠️  无标签剔除数据，跳过生成热力图")
        
#         # 4. 主目录 CSV（含各标签剔除明细）
#         output_filename_csv = f"{dataset_name}_valid_statistics.csv"
#         output_path_csv = os.path.join(output_dir, output_filename_csv)
#         write_detailed_csv(
#             output_path_csv, os.path.basename(directory), dir_label_counts,
#             dir_pic_counts, dir_subfolder_counts,
#             cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all,
#             reject_stats, dir_raw_count,
#             reject_by_label_stats, dir_label_raw_counts,  # 【新增】
#             subfolder_summary_list=subfolder_stats_collection,
#             filter_info=filter_info
#         )
#         print(f"✅ 主目录 CSV 统计文件已保存至：{output_path_csv}")
    
        
#         # 5. 子文件夹独立统计（不生成热力图，只生成主目录的）
#         print(f"\n   正在生成 {len(subfolder_stats_collection)} 个子文件夹的独立统计...")
#         for idx, sub_info in enumerate(subfolder_stats_collection, 1):
#             safe_sub_name = sub_info['path'].replace(os.sep, '_').replace('.', 'root')
            
#             # 5.1 子文件夹 CSV
#             sub_csv_name = f"subfolder_{safe_sub_name}_stats.csv"
#             sub_csv_path = os.path.join(output_dir, sub_csv_name)
#             write_detailed_csv(
#                 sub_csv_path,
#                 f"{os.path.basename(directory)} / {sub_info['path']}",
#                 sub_info['label_counts'],
#                 sub_info['pic_counts'], 1,
#                 sub_info['cat0'], sub_info['cat1'], sub_info['cat2'], 
#                 sub_info['undef'], sub_info['cat1_pct'], sub_info['total_all'],
#                 sub_info['reject_stats'], sub_info['raw_count'],
#                 {}, {},  # 子文件夹不生成按标签剔除统计
#                 subfolder_summary_list=None, filter_info=filter_info
#             )
            
#             # 5.2 子文件夹 - 有效分布图
#             sub_png_name = f"subfolder_{safe_sub_name}_distribution.png"
#             sub_png_path = os.path.join(output_dir, sub_png_name)
#             create_distribution_plot(
#                 sub_info['label_counts'], sub_png_path,
#                 f"{os.path.basename(directory)} / {sub_info['path']}",
#                 sub_info['cat1'], sub_info['total_all'],
#                 sub_info['cat1_pct'], filter_info=filter_info
#             )
            
#             # 5.3 子文件夹 - 剔除原因图
#             sub_reject_png = f"subfolder_{safe_sub_name}_rejection.png"
#             sub_reject_path = os.path.join(output_dir, sub_reject_png)
#             create_rejection_plot(
#                 sub_info['reject_stats'], sub_info['raw_count'],
#                 sub_reject_path, f"{os.path.basename(directory)} / {sub_info['path']}",
#                 filter_info=filter_info
#             )
            
#             print(f"   [{idx}/{len(subfolder_stats_collection)}] {sub_info['path']} - CSV & PNGs 已生成")
        
#         print(f"   ✅ 所有子文件夹统计生成完成。")

#     else:
#         print(f"\n⚠ 警告：该目录下未找到任何有效标注数据")

# # ============ 全局汇总 ============
# print(f"\n{'='*60}")
# print("正在生成所有目录的汇总统计 CSV...")
# print(f"{'='*60}")

# all_datasets_stats = []
# total_images_all = 0
# total_raw_all = 0
# total_cat0_all = 0
# total_cat1_all = 0
# total_cat2_all = 0
# total_undefined_all = 0
# total_reject_area_all = 0
# total_reject_circ_all = 0
# total_reject_edge_all = 0

# for directory in directories:
#     dir_label_counts = collections.defaultdict(int)
#     dir_reject_counts = collections.defaultdict(int)
#     dir_reject_counts_by_label = collections.defaultdict(lambda: collections.defaultdict(int))
#     dir_label_raw_counts = collections.defaultdict(int)
#     dir_raw_count = 0
#     dir_pic_counts = 0
    
#     for root, dirs, files in os.walk(directory):
#         for filename in files:
#             if filename.endswith(".json"):
#                 file_path = os.path.join(root, filename)
#                 dir_pic_counts += 1
                
#                 img_path = find_matching_image(file_path, image_shape_cache)
#                 img_shape = get_image_shape(img_path, image_shape_cache)
                
#                 try:
#                     with open(file_path, "r", encoding="utf-8") as f:
#                         data = json.load(f)
#                     for shape in data["shapes"]:
#                         label = shape["label"]
#                         points = shape.get("points")
#                         dir_raw_count += 1
#                         dir_label_raw_counts[label] += 1
                        
#                         if img_shape and points:
#                             is_valid, reject_reason = is_valid_cell(
#                                 points, img_shape, FILTER_EDGE_CELLS, EDGE_MARGIN, MIN_AREA, MIN_CIRCULARITY
#                             )
#                             if is_valid:
#                                 dir_label_counts[label] += 1
#                             else:
#                                 dir_reject_counts[reject_reason] += 1
#                                 dir_reject_counts_by_label[label][reject_reason] += 1
#                         else:
#                             dir_reject_counts["invalid_input"] += 1
#                             dir_reject_counts_by_label[label]["invalid_input"] += 1
#                 except Exception as e:
#                     dir_reject_counts["exception"] += 1
    
#     cat0, cat1, cat2, undefined, cat1_pct, total_all = calculate_category_stats(dir_label_counts, cell_dict_big)
#     reject_stats = calculate_reject_stats(dir_reject_counts, dir_raw_count)
    
#     all_datasets_stats.append({
#         'dataset': os.path.basename(directory),
#         'images': dir_pic_counts,
#         'raw_annotations': dir_raw_count,
#         'valid_cells': total_all,
#         'cat0_count': cat0,
#         'cat1_count': cat1,
#         'cat2_count': cat2,
#         'undefined_count': undefined,
#         'total_all': total_all,
#         'cat1_percentage': cat1_pct,
#         'reject_area': reject_stats['area']['count'],
#         'reject_circ': reject_stats['circularity']['count'],
#         'reject_edge': reject_stats['edge']['count']
#     })
#     total_images_all += dir_pic_counts
#     total_raw_all += dir_raw_count
#     total_cat0_all += cat0
#     total_cat1_all += cat1
#     total_cat2_all += cat2
#     total_undefined_all += undefined
#     total_reject_area_all += reject_stats['area']['count']
#     total_reject_circ_all += reject_stats['circularity']['count']
#     total_reject_edge_all += reject_stats['edge']['count']

# global_output_dir = os.path.dirname(directories[0])
# if not os.path.exists(global_output_dir):
#     global_output_dir = "./global_statistics"
#     os.makedirs(global_output_dir, exist_ok=True)

# global_csv_path = os.path.join(GLOBAL_OUTPUT_ROOT, "ALL_DATASETS_VALID_SUMMARY.csv")
# with open(global_csv_path, "w", encoding="utf-8-sig", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["=== Global Summary (Valid Cells Only) ==="])
#     writer.writerow(["Filter Settings", f"Area>={MIN_AREA}, Circ>={MIN_CIRCULARITY}, Margin={EDGE_MARGIN}"])
#     writer.writerow(["Total Datasets", len(directories)])
#     writer.writerow(["Total Images", total_images_all])
#     writer.writerow(["Total Raw Annotations", total_raw_all])
#     writer.writerow(["Total Valid Cells", total_cat0_all + total_cat1_all + total_cat2_all + total_undefined_all])
#     writer.writerow(["Total Rejected Cells", total_raw_all - (total_cat0_all + total_cat1_all + total_cat2_all + total_undefined_all)])
#     writer.writerow([])
#     writer.writerow(["=== Rejection Reasons (Global) ==="])
#     writer.writerow(["Reason", "Count", "Percentage of Raw"])
#     writer.writerow(["Area < threshold", total_reject_area_all, f"{(total_reject_area_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
#     writer.writerow(["Circularity < threshold", total_reject_circ_all, f"{(total_reject_circ_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
#     writer.writerow(["Edge margin", total_reject_edge_all, f"{(total_reject_edge_all/total_raw_all*100) if total_raw_all>0 else 0:.2f}%"])
#     writer.writerow([])
#     writer.writerow(["Dataset", "Images", "Raw Annotations", "Valid Cells", "Rejected", "Cat-1 Count", "Cat-1 %", "Reject(Area)", "Reject(Circ)", "Reject(Edge)"])
#     for ds in all_datasets_stats:
#         rejected = ds['raw_annotations'] - ds['valid_cells']
#         writer.writerow([
#             ds['dataset'], ds['images'], ds['raw_annotations'], ds['valid_cells'], rejected,
#             ds['cat1_count'], f"{ds['cat1_percentage']:.2f}%",
#             ds['reject_area'], ds['reject_circ'], ds['reject_edge']
#         ])

# print(f"\n{'='*60}")
# print("✅ 所有目录处理完成！")
# print(f"✅ 全局汇总 CSV 已保存至：{global_csv_path}")
# print(f"{'='*60}")