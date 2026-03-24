import os
import json
import collections
import matplotlib.pyplot as plt
import csv

'''
统计每个指定文件夹的总标签分布（汇总该文件夹下所有子文件夹）
并根据 cell_dict_big 映射统计类别 1 细胞占比
分母包含所有细胞（包括未定义标签）
新增：为每个子文件夹也生成统计图
'''

# 指定要遍历的目录
directories = [
    "/root/autodl-tmp/data/MAIN_imgs_260323"
    "/root/autodl-tmp/data/FXH_imgs_noALL_260318",
    "/root/autodl-tmp/data/BJH_imgs_260211",
    "/root/autodl-tmp/data/TJMU_imgs_260318",
    # "/root/autodl-tmp/data/MAIN_imgs_outline_mask_260211",
    # "/root/autodl-tmp/data/MAIN_imgs_dot_mask_260211",
]

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

def calculate_category_stats(label_counts, cell_dict_big):
    """
    根据 cell_dict_big 映射计算各类别统计
    分母包含所有细胞（包括未定义标签）
    """
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

def write_detailed_csv(output_path, dataset_name, label_counts, dir_pic_counts, dir_subfolder_counts, 
                       cat0, cat1, cat2, undefined_count, cat1_pct, total_all, subfolder_summary_list=None):
    """
    写入详细的 CSV 统计文件
    """
    with open(output_path, "w", encoding="utf-8-sig", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 1. 文件头信息
        writer.writerow(["Dataset", dataset_name])
        writer.writerow(["Total Images", dir_pic_counts])
        writer.writerow(["Total Subfolders", dir_subfolder_counts])
        writer.writerow(["Total Annotations", sum(label_counts.values())])
        writer.writerow(["Total Cells (All Labels)", total_all])
        writer.writerow([])
        
        # 2. 类别统计汇总
        writer.writerow(["=== Category Statistics (by cell_dict_big) ==="])
        writer.writerow(["Category", "Description", "Count", "Percentage of Total Cells"])
        
        total_for_pct = total_all if total_all > 0 else 1
        writer.writerow(["0", "Background (V, 0)", cat0, f"{cat0/total_for_pct*100:.2f}%"])
        writer.writerow(["1", "Target Cells (N, N1, M, M1, R, R1, J, J1)", cat1, f"{cat1_pct:.2f}%"])
        writer.writerow(["2", "Other Cells", cat2, f"{cat2/total_for_pct*100:.2f}%"])
        
        # 具体列出未定义标签明细
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
        
        # 3. 各标签详细统计 (已知标签)
        writer.writerow(["=== Label Details (Defined) ==="])
        writer.writerow(["Label", "Order", "BigCategory", "Count", "Percentage of Total"])
        
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
        
        # 如果是主目录 CSV，添加子文件夹统计摘要
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

def create_distribution_plot(label_counts, output_path, title_prefix, cat1_count, total_all, cat1_pct):
    """
    创建并保存标签分布柱状图
    """
    plt.figure(figsize=(14, 7))
    counts = [label_counts.get(label, 0) for label in sorted_labels]
    bars = plt.bar(sorted_labels, counts, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # 根据类别设置颜色
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
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.title(f"Total Label Distribution - {title_prefix}\n"
              f"Category-1 Ratio: {cat1_pct:.2f}% ({cat1_count}/{total_all})", 
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
    
    # 在柱体上方标记数值
    for i, (label, count) in enumerate(zip(sorted_labels, counts)):
        if count > 0:
            plt.text(i, count, str(count), ha='center', va='bottom', 
                    fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# 遍历每个主目录
for directory in directories:
    print(f"\n{'='*60}")
    print(f"正在处理目录：{directory}")
    print(f"{'='*60}")
    
    # 当前目录的全局统计
    dir_label_counts = collections.defaultdict(int)
    dir_pic_counts = 0
    dir_subfolder_counts = 0
    
    # 存储子文件夹统计信息的列表
    subfolder_stats_collection = [] 
    
    # 遍历目录下的所有子文件夹
    for root, dirs, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        
        # 初始化当前子文件夹的统计
        subfolder_label_counts = collections.defaultdict(int)
        subfolder_pic_counts = 0
        
        # 遍历当前子文件夹中的文件
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                subfolder_pic_counts += 1
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue
                        label = shape["label"]
                        subfolder_label_counts[label] += 1
                        dir_label_counts[label] += 1
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")
        
        # 如果子文件夹有数据，记录其统计信息
        if subfolder_pic_counts > 0:
            dir_pic_counts += subfolder_pic_counts
            dir_subfolder_counts += 1
            
            # 计算当前子文件夹的类别统计
            s_cat0, s_cat1, s_cat2, s_undef, s_cat1_pct, s_total = calculate_category_stats(
                subfolder_label_counts, cell_dict_big
            )
            
            # 存入列表（键名统一）
            subfolder_stats_collection.append({
                'path': relative_path,
                'label_counts': dict(subfolder_label_counts),
                'pic_counts': subfolder_pic_counts,
                'cat0': s_cat0,
                'cat1': s_cat1,
                'cat2': s_cat2,
                'undef': s_undef,
                'cat1_pct': s_cat1_pct,
                'total_all': s_total
            })
    
    # 生成当前目录的总分布图和 CSV
    if dir_pic_counts > 0:
        # 计算主目录的类别统计
        cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all = calculate_category_stats(
            dir_label_counts, cell_dict_big
        )
        
        print(f"\n📊 目录统计摘要:")
        print(f"   子文件夹数：{dir_subfolder_counts}")
        print(f"   图片总数：{dir_pic_counts}")
        print(f"   🎯 类别 1 细胞占比：{cat1_percentage:.2f}%")
        
        # 创建保存目录
        output_dir = "/root/autodl-tmp/data/statistics_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # ========== 1. 绘制并保存主目录柱状图 ==========
        output_filename_png = f"{os.path.basename(directory)}_total_distribution.png"
        output_path_png = os.path.join(output_dir, output_filename_png)
        create_distribution_plot(
            dir_label_counts, 
            output_path_png, 
            os.path.basename(directory),
            cat1_count, 
            total_all, 
            cat1_percentage
        )
        print(f"\n✅ 主目录图表已保存至：{output_path_png}")
        
        # ========== 2. 生成主目录 CSV 统计文件 (包含子文件夹摘要) ==========
        output_filename_csv = f"{os.path.basename(directory)}_total_statistics.csv"
        output_path_csv = os.path.join(output_dir, output_filename_csv)
        
        write_detailed_csv(
            output_path_csv, 
            os.path.basename(directory), 
            dir_label_counts, 
            dir_pic_counts, 
            dir_subfolder_counts,
            cat0_count, cat1_count, cat2_count, undefined_count, cat1_percentage, total_all,
            subfolder_summary_list=subfolder_stats_collection
        )
        print(f"✅ 主目录 CSV 统计文件已保存至：{output_path_csv}")
        
        # ========== 3. 为每个子文件夹生成独立 CSV 和统计图 ==========
        print(f"\n   正在生成 {len(subfolder_stats_collection)} 个子文件夹的独立统计 CSV 和图表...")
        for idx, sub_info in enumerate(subfolder_stats_collection, 1):
            safe_sub_name = sub_info['path'].replace(os.sep, '_').replace('.', 'root')
            
            # --- 生成子文件夹 CSV ---
            sub_csv_name = f"subfolder_{safe_sub_name}_stats.csv"
            sub_csv_path = os.path.join(output_dir, sub_csv_name)
            
            write_detailed_csv(
                sub_csv_path,
                f"{os.path.basename(directory)} / {sub_info['path']}",
                sub_info['label_counts'],
                sub_info['pic_counts'],
                1,
                sub_info['cat0'], 
                sub_info['cat1'],
                sub_info['cat2'], 
                sub_info['undef'],
                sub_info['cat1_pct'],
                sub_info['total_all'],
                subfolder_summary_list=None
            )
            
            # --- 生成子文件夹统计图 ---
            sub_png_name = f"subfolder_{safe_sub_name}_distribution.png"
            sub_png_path = os.path.join(output_dir, sub_png_name)
            create_distribution_plot(
                sub_info['label_counts'],
                sub_png_path,
                f"{os.path.basename(directory)} / {sub_info['path']}",
                sub_info['cat1'],
                sub_info['total_all'],
                sub_info['cat1_pct']
            )
            
            print(f"   [{idx}/{len(subfolder_stats_collection)}] {sub_info['path']} - CSV & PNG 已生成")
        
        print(f"   ✅ 所有子文件夹 CSV 和图表生成完成。")

    else:
        print(f"\n⚠ 警告：该目录下未找到任何有效标注数据")

# ============ 全局汇总 ============
print(f"\n{'='*60}")
print("正在生成所有目录的汇总统计 CSV...")
print(f"{'='*60}")

all_datasets_stats = []
total_images_all = 0
total_annotations_all = 0
total_cat0_all = 0
total_cat1_all = 0
total_cat2_all = 0
total_undefined_all = 0

for directory in directories:
    dir_label_counts = collections.defaultdict(int)
    dir_pic_counts = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                dir_pic_counts += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for shape in data["shapes"]:
                        if shape.get("shape_type") != "polygon":
                            continue
                        label = shape["label"]
                        dir_label_counts[label] += 1
                except Exception as e:
                    pass
    
    cat0, cat1, cat2, undefined, cat1_pct, total_all = calculate_category_stats(dir_label_counts, cell_dict_big)
    
    all_datasets_stats.append({
        'dataset': os.path.basename(directory),
        'images': dir_pic_counts,
        'annotations': sum(dir_label_counts.values()),
        'cat0_count': cat0,
        'cat1_count': cat1,
        'cat2_count': cat2,
        'undefined_count': undefined,
        'total_all': total_all,
        'cat1_percentage': cat1_pct
    })
    total_images_all += dir_pic_counts
    total_annotations_all += sum(dir_label_counts.values())
    total_cat0_all += cat0
    total_cat1_all += cat1
    total_cat2_all += cat2
    total_undefined_all += undefined

# 保存全局汇总 CSV
global_output_dir = os.path.dirname(directories[0])
if not os.path.exists(global_output_dir):
    global_output_dir = "./global_statistics"
    os.makedirs(global_output_dir, exist_ok=True)

global_csv_path = os.path.join(global_output_dir, "ALL_DATASETS_SUMMARY.csv")
with open(global_csv_path, "w", encoding="utf-8-sig", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["=== Global Summary ==="])
    writer.writerow(["Total Datasets", len(directories)])
    writer.writerow(["Total Images", total_images_all])
    writer.writerow(["Total Annotations", total_annotations_all])
    writer.writerow(["Total Cells", total_cat0_all + total_cat1_all + total_cat2_all + total_undefined_all])
    writer.writerow([])
    writer.writerow(["Dataset", "Images", "Total Cells", "Cat-1 Count", "Cat-1 %"])
    for ds in all_datasets_stats:
        writer.writerow([
            ds['dataset'], ds['images'], ds['total_all'], ds['cat1_count'], f"{ds['cat1_percentage']:.2f}%"
        ])

print(f"\n{'='*60}")
print("✅ 所有目录处理完成！")
print(f"✅ 全局汇总 CSV 已保存至：{global_csv_path}")
print(f"{'='*60}")