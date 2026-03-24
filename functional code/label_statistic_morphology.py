import os
import json
import collections
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import math

'''
统计每个文件夹内所有细胞的圆度和面积分布
按细胞类型给出分布直方图，并将分布结果记录在表格中
'''

# ================= 配置区域 =================
# 指定要遍历的目录
directories = [
    "/root/autodl-tmp/data/MAIN_imgs_260312"
]

GLOBAL_OUTPUT_ROOT = "/root/autodl-tmp/data/statistics_morphology"
os.makedirs(GLOBAL_OUTPUT_ROOT, exist_ok=True)

# 标签顺序定义
cell_dict = {
    "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
    "E": 8, "B": 9, "M0": 10, "M": 11, "M1": 12, "M2": 13,
    "R": 14, "R1": 15, "R2": 16, "R3": 17,
    "J": 18, "J1": 19, "J2": 20, "J3": 21, "J4": 22,
    "L": 23, "L1": 24, "L2": 25, "L3": 26, "L4": 27,
    "P": 28, "P1": 29, "P2": 30, "P3": 31,
    "B1": 32, "E1": 33, "A": 34, "F": 35, "V": 36, "0": 36
}

sorted_labels = sorted(cell_dict.keys(), key=lambda x: cell_dict[x])

# ================= 辅助函数 =================

def calculate_morphology(points, image_shape):
    """
    计算细胞的形态学特征
    Returns:
        (area, circularity) or (None, None) if invalid
    """
    if not points or image_shape is None:
        return None, None
    
    try:
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        contour = np.ascontiguousarray(contour)
        
        area = cv2.contourArea(contour)
        if area <= 0:
            return None, None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None, None
        
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        
        return area, circularity
    except Exception:
        return None, None

def get_image_shape(image_path, cache):
    """获取图像尺寸，带缓存"""
    if image_path in cache:
        return cache[image_path]
    
    if not os.path.exists(image_path):
        return None
    
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    cache[image_path] = (h, w)
    return (h, w)

def find_matching_image(json_path):
    """根据 JSON 路径寻找对应的图像文件"""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    parent_dir = os.path.dirname(json_path)
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG']:
        potential_path = os.path.join(parent_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None

def create_morphology_histograms(label_morphology_data, output_dir, dataset_name, filter_info=None):
    """
    为每个细胞类型创建面积和圆度分布直方图
    label_morphology_data: {label: [(area, circ), ...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 只处理有数据的标签
    labels_with_data = [l for l in sorted_labels if l in label_morphology_data and len(label_morphology_data[l]) > 0]
    
    if len(labels_with_data) == 0:
        return None
    
    # 1. 面积分布直方图（所有标签合并）
    plt.figure(figsize=(14, 8))
    all_areas = []
    for label in labels_with_data:
        areas = [m[0] for m in label_morphology_data[label] if m[0] is not None]
        all_areas.extend(areas)
    
    if len(all_areas) > 0:
        plt.hist(all_areas, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Cell Area (pixels²)", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.title(f"Area Distribution - {dataset_name}\nTotal Cells: {len(all_areas)}", 
                  fontsize=13, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.axvline(np.mean(all_areas), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_areas):.1f}')
        plt.axvline(np.median(all_areas), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_areas):.1f}')
        plt.legend()
        plt.tight_layout()
        area_hist_path = os.path.join(output_dir, f"{dataset_name}_area_distribution.png")
        plt.savefig(area_hist_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        area_hist_path = None
    
    # 2. 圆度分布直方图（所有标签合并）
    plt.figure(figsize=(14, 8))
    all_circs = []
    for label in labels_with_data:
        circs = [m[1] for m in label_morphology_data[label] if m[1] is not None]
        all_circs.extend(circs)
    
    if len(all_circs) > 0:
        plt.hist(all_circs, bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.xlabel("Circularity", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.title(f"Circularity Distribution - {dataset_name}\nTotal Cells: {len(all_circs)}", 
                  fontsize=13, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.axvline(np.mean(all_circs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_circs):.3f}')
        plt.axvline(np.median(all_circs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_circs):.3f}')
        plt.xlim(0, 1.2)
        plt.legend()
        plt.tight_layout()
        circ_hist_path = os.path.join(output_dir, f"{dataset_name}_circularity_distribution.png")
        plt.savefig(circ_hist_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        circ_hist_path = None
    
    # 3. 按细胞类型的面积分布（多子图）
    n_labels = len(labels_with_data)
    if n_labels > 0:
        cols = 5
        rows = (n_labels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, label in enumerate(labels_with_data):
            areas = [m[0] for m in label_morphology_data[label] if m[0] is not None]
            if len(areas) > 0:
                axes[idx].hist(areas, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f"{label}\n(n={len(areas)}, mean={np.mean(areas):.1f})", fontsize=10, fontweight='bold')
                axes[idx].set_xlabel("Area", fontsize=8)
                axes[idx].set_ylabel("Freq", fontsize=8)
                axes[idx].grid(axis='y', linestyle='--', alpha=0.3)
            else:
                axes[idx].axis('off')
        
        for idx in range(n_labels, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f"Area Distribution by Cell Type - {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        area_by_label_path = os.path.join(output_dir, f"{dataset_name}_area_by_label.png")
        plt.savefig(area_by_label_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        area_by_label_path = None
    
    # 4. 按细胞类型的圆度分布（多子图）
    if n_labels > 0:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, label in enumerate(labels_with_data):
            circs = [m[1] for m in label_morphology_data[label] if m[1] is not None]
            if len(circs) > 0:
                axes[idx].hist(circs, bins=30, color='coral', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f"{label}\n(n={len(circs)}, mean={np.mean(circs):.3f})", fontsize=10, fontweight='bold')
                axes[idx].set_xlabel("Circularity", fontsize=8)
                axes[idx].set_ylabel("Freq", fontsize=8)
                axes[idx].grid(axis='y', linestyle='--', alpha=0.3)
                axes[idx].set_xlim(0, 1.2)
            else:
                axes[idx].axis('off')
        
        for idx in range(n_labels, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f"Circularity Distribution by Cell Type - {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        circ_by_label_path = os.path.join(output_dir, f"{dataset_name}_circularity_by_label.png")
        plt.savefig(circ_by_label_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        circ_by_label_path = None
    
    return {
        'area_hist': area_hist_path,
        'circ_hist': circ_hist_path,
        'area_by_label': area_by_label_path,
        'circ_by_label': circ_by_label_path
    }

def write_morphology_csv(output_path, dataset_name, label_morphology_data, total_cells, filter_info=None):
    """写入形态学统计 CSV 文件"""
    with open(output_path, "w", encoding="utf-8-sig", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 1. 文件头信息
        writer.writerow(["Dataset", dataset_name])
        writer.writerow(["Total Cells", total_cells])
        if filter_info:
            writer.writerow(["Note", "All cells included (no filtering)"])
        writer.writerow([])
        
        # 2. 总体统计
        all_areas = []
        all_circs = []
        for label in sorted_labels:
            if label in label_morphology_data:
                for area, circ in label_morphology_data[label]:
                    if area is not None:
                        all_areas.append(area)
                    if circ is not None:
                        all_circs.append(circ)
        
        writer.writerow(["=== Overall Statistics ==="])
        writer.writerow(["Metric", "Area", "Circularity"])
        if len(all_areas) > 0:
            writer.writerow(["Count", len(all_areas), len(all_circs)])
            writer.writerow(["Mean", f"{np.mean(all_areas):.2f}", f"{np.mean(all_circs):.4f}"])
            writer.writerow(["Median", f"{np.median(all_areas):.2f}", f"{np.median(all_circs):.4f}"])
            writer.writerow(["Std", f"{np.std(all_areas):.2f}", f"{np.std(all_circs):.4f}"])
            writer.writerow(["Min", f"{np.min(all_areas):.2f}", f"{np.min(all_circs):.4f}"])
            writer.writerow(["Max", f"{np.max(all_areas):.2f}", f"{np.max(all_circs):.4f}"])
            writer.writerow(["Q1 (25%)", f"{np.percentile(all_areas, 25):.2f}", f"{np.percentile(all_circs, 25):.4f}"])
            writer.writerow(["Q3 (75%)", f"{np.percentile(all_areas, 75):.2f}", f"{np.percentile(all_circs, 75):.4f}"])
        writer.writerow([])
        
        # 3. 按细胞类型统计
        writer.writerow(["=== Statistics by Cell Type ==="])
        writer.writerow(["Label", "Count", 
                        "Area_Mean", "Area_Median", "Area_Std", "Area_Min", "Area_Max",
                        "Circ_Mean", "Circ_Median", "Circ_Std", "Circ_Min", "Circ_Max"])
        
        for label in sorted_labels:
            if label in label_morphology_data and len(label_morphology_data[label]) > 0:
                morph_list = label_morphology_data[label]
                areas = [m[0] for m in morph_list if m[0] is not None]
                circs = [m[1] for m in morph_list if m[1] is not None]
                
                if len(areas) > 0 and len(circs) > 0:
                    writer.writerow([
                        label,
                        len(morph_list),
                        f"{np.mean(areas):.2f}", f"{np.median(areas):.2f}", f"{np.std(areas):.2f}", 
                        f"{np.min(areas):.2f}", f"{np.max(areas):.2f}",
                        f"{np.mean(circs):.4f}", f"{np.median(circs):.4f}", f"{np.std(circs):.4f}",
                        f"{np.min(circs):.4f}", f"{np.max(circs):.4f}"
                    ])
        
        writer.writerow([])
        
        # 4. 分布区间统计（面积）
        writer.writerow(["=== Area Distribution Intervals ==="])
        writer.writerow(["Interval (pixels²)", "Count", "Percentage"])
        
        if len(all_areas) > 0:
            intervals = [
                (0, 5000), (5000, 10000), (10000, 20000), (20000, 30000),
                (30000, 50000), (50000, 80000), (80000, 100000), (100000, float('inf'))
            ]
            for low, high in intervals:
                count = sum(1 for a in all_areas if low <= a < high)
                pct = (count / len(all_areas) * 100) if len(all_areas) > 0 else 0
                high_str = f"{high:.0f}" if high != float('inf') else "∞"
                writer.writerow([f"[{low:.0f}, {high_str})", count, f"{pct:.2f}%"])
        
        writer.writerow([])
        
        # 5. 分布区间统计（圆度）
        writer.writerow(["=== Circularity Distribution Intervals ==="])
        writer.writerow(["Interval", "Count", "Percentage"])
        
        if len(all_circs) > 0:
            circ_intervals = [
                (0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                (0.8, 0.9), (0.9, 1.0), (1.0, 1.2)
            ]
            for low, high in circ_intervals:
                count = sum(1 for c in all_circs if low <= c < high)
                pct = (count / len(all_circs) * 100) if len(all_circs) > 0 else 0
                writer.writerow([f"[{low:.1f}, {high:.1f})", count, f"{pct:.2f}%"])

# ================= 主处理流程 =================

image_shape_cache = {}

for directory in directories:
    print(f"\n{'='*60}")
    print(f"正在处理目录：{directory}")
    print(f"{'='*60}")
    
    # 存储每个标签的形态学数据: {label: [(area, circ), ...]}
    label_morphology_data = collections.defaultdict(list)
    total_cells = 0
    dir_pic_counts = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                dir_pic_counts += 1
                
                # 寻找对应的图像文件
                img_path = find_matching_image(file_path)
                img_shape = get_image_shape(img_path, image_shape_cache)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue
                        
                        label = shape["label"]
                        points = shape.get("points")
                        
                        # 计算形态学特征
                        area, circularity = calculate_morphology(points, img_shape)
                        
                        if area is not None and circularity is not None:
                            label_morphology_data[label].append((area, circularity))
                            total_cells += 1
                        else:
                            # 无效标注也计数（但形态特征为 None）
                            label_morphology_data[label].append((None, None))
                            total_cells += 1
                            
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")
    
    if total_cells > 0:
        dataset_name = os.path.basename(directory)
        output_dir = os.path.join(GLOBAL_OUTPUT_ROOT, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 目录统计摘要:")
        print(f"   图片总数：{dir_pic_counts}")
        print(f"   细胞总数：{total_cells}")
        print(f"   有形态数据的标签数：{len([l for l in label_morphology_data if len(label_morphology_data[l]) > 0])}")
        
        # 1. 创建形态学分布直方图
        print(f"\n   正在生成分布直方图...")
        plot_paths = create_morphology_histograms(
            label_morphology_data, output_dir, dataset_name
        )
        
        if plot_paths:
            if plot_paths.get('area_hist'):
                print(f"   ✅ 面积分布图：{plot_paths['area_hist']}")
            if plot_paths.get('circ_hist'):
                print(f"   ✅ 圆度分布图：{plot_paths['circ_hist']}")
            if plot_paths.get('area_by_label'):
                print(f"   ✅ 按标签面积图：{plot_paths['area_by_label']}")
            if plot_paths.get('circ_by_label'):
                print(f"   ✅ 按标签圆度图：{plot_paths['circ_by_label']}")
        
        # 2. 写入形态学统计 CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}_morphology_statistics.csv")
        write_morphology_csv(csv_path, dataset_name, label_morphology_data, total_cells)
        print(f"\n   ✅ 形态学统计 CSV：{csv_path}")
        
    else:
        print(f"\n⚠ 警告：该目录下未找到任何有效细胞标注数据")

# ============ 全局汇总 ============
print(f"\n{'='*60}")
print("正在生成所有目录的全局汇总统计...")
print(f"{'='*60}")

global_all_areas = []
global_all_circs = []
global_summary = []

for directory in directories:
    label_morphology_data = collections.defaultdict(list)
    total_cells = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                img_path = find_matching_image(file_path)
                img_shape = get_image_shape(img_path, image_shape_cache)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue
                        label = shape["label"]
                        points = shape.get("points")
                        area, circularity = calculate_morphology(points, img_shape)
                        
                        if area is not None and circularity is not None:
                            label_morphology_data[label].append((area, circularity))
                            total_cells += 1
                            global_all_areas.append(area)
                            global_all_circs.append(circularity)
                except Exception:
                    pass
    
    # 计算该数据集的统计
    areas = [m[0] for m_list in label_morphology_data.values() for m in m_list if m[0] is not None]
    circs = [m[1] for m_list in label_morphology_data.values() for m in m_list if m[1] is not None]
    
    global_summary.append({
        'dataset': os.path.basename(directory),
        'cells': total_cells,
        'area_mean': np.mean(areas) if areas else 0,
        'area_median': np.median(areas) if areas else 0,
        'circ_mean': np.mean(circs) if circs else 0,
        'circ_median': np.median(circs) if circs else 0
    })

# 写入全局汇总 CSV
global_csv_path = os.path.join(GLOBAL_OUTPUT_ROOT, "ALL_DATASETS_MORPHOLOGY_SUMMARY.csv")
with open(global_csv_path, "w", encoding="utf-8-sig", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(["=== Global Morphology Summary ==="])
    writer.writerow(["Total Datasets", len(directories)])
    writer.writerow(["Total Cells", len(global_all_areas)])
    writer.writerow([])
    
    writer.writerow(["=== Global Statistics ==="])
    writer.writerow(["Metric", "Area", "Circularity"])
    if len(global_all_areas) > 0:
        writer.writerow(["Count", len(global_all_areas), len(global_all_circs)])
        writer.writerow(["Mean", f"{np.mean(global_all_areas):.2f}", f"{np.mean(global_all_circs):.4f}"])
        writer.writerow(["Median", f"{np.median(global_all_areas):.2f}", f"{np.median(global_all_circs):.4f}"])
        writer.writerow(["Std", f"{np.std(global_all_areas):.2f}", f"{np.std(global_all_circs):.4f}"])
        writer.writerow(["Min", f"{np.min(global_all_areas):.2f}", f"{np.min(global_all_circs):.4f}"])
        writer.writerow(["Max", f"{np.max(global_all_areas):.2f}", f"{np.max(global_all_circs):.4f}"])
    writer.writerow([])
    
    writer.writerow(["=== Dataset Summary ==="])
    writer.writerow(["Dataset", "Cells", "Area_Mean", "Area_Median", "Circ_Mean", "Circ_Median"])
    for ds in global_summary:
        writer.writerow([
            ds['dataset'], ds['cells'],
            f"{ds['area_mean']:.2f}", f"{ds['area_median']:.2f}",
            f"{ds['circ_mean']:.4f}", f"{ds['circ_median']:.4f}"
        ])

print(f"\n{'='*60}")
print("✅ 所有目录处理完成！")
print(f"✅ 全局汇总 CSV：{global_csv_path}")
print(f"✅ 所有输出目录：{GLOBAL_OUTPUT_ROOT}")
print(f"{'='*60}")