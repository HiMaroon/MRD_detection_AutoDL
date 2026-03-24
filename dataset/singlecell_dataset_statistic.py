# -*- coding: utf-8 -*-
import os
import re
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import csv

'''
Batch statistics for image category distribution in multiple folders
Image naming format: xxx_xxx_LABEL.png (label is after the last underscore)
Generate independent statistical files for each input folder and global summary
Undefined labels are categorized as Class 0 (using "0" label)
'''

# ==================== Configuration Area ====================

# List of folder paths to be statistics
TARGET_DIRS = [
    "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/train_groundtruth",
    "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/val_groundtruth",
    "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/train",
    "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/val",
]

# Output directory (all statistical results stored together)
OUTPUT_DIR = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/singlecell/statistics"

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Cell type mapping dictionary (for ordering)
# Note: Labels not in the dictionary will be automatically categorized as "0"
cell_dict = {
    "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
    "E": 8, "B": 9, "M0": 10, "M": 11, "M1": 12, "M2": 13,
    "R": 14, "R1": 15, "R2": 16, "R3": 17,
    "J": 18, "J1": 19, "J2": 20, "J3": 21, "J4": 22,
    "L": 23, "L1": 24, "L2": 25, "L3": 26, "L4": 27,
    "P": 28, "P1": 29, "P2": 30, "P3": 31,
    "B1": 32, "E1": 33, "A": 34, "F": 35, "V": 36, "0": 36
}

# Get all unique keys from dictionary in the specified order
# Sort by the value in cell_dict to maintain the custom order
DICT_KEYS = sorted(cell_dict.keys(), key=lambda x: cell_dict[x])

# ==================== Core Functions ====================

def extract_label_from_filename(filename):
    """
    Extract category label from filename
    Format: xxx_xxx_LABEL.png -> Returns "LABEL"
    """
    name_no_ext = Path(filename).stem
    parts = name_no_ext.rsplit('_', 1)
    if len(parts) >= 2:
        return parts[-1]
    return None

def collect_image_stats(directory, recursive=True):
    """
    Traverse directory to collect image label statistics
    """
    label_counts = collections.defaultdict(int)
    file_list = []
    dir_path = Path(directory)
    
    if recursive:
        files = dir_path.rglob("*")
    else:
        files = dir_path.iterdir()
    
    for file_path in files:
        if file_path.suffix.lower() in IMAGE_EXTENSIONS and file_path.is_file():
            label = extract_label_from_filename(file_path.name)
            if label:
                # If label is not in dictionary, categorize as "0"
                if label not in cell_dict:
                    label = "0"
                label_counts[label] += 1
                file_list.append((str(file_path), label))
    
    return dict(label_counts), sum(label_counts.values()), file_list

def get_label_color(label):
    """
    Get color for each label (using a colormap for better visualization)
    """
    # Use a colormap to generate distinct colors for different labels
    cmap = plt.cm.tab20
    if label in cell_dict:
        # Use the index from cell_dict to determine color
        idx = cell_dict[label] % 20
        return cmap(idx)
    return '#808080'  # Gray for undefined

def create_bar_chart(label_counts, output_path, title, dict_keys, total_count=None):
    """
    Create bar chart with x-axis showing only dictionary keys in specified order
    """
    # Show all keys from dictionary in the specified order
    all_keys = dict_keys
    counts = []
    colors = []
    
    for key in all_keys:
        count = label_counts.get(key, 0)
        counts.append(count)
        colors.append(get_label_color(key))
    
    total = total_count or sum(counts)
    
    plt.figure(figsize=(max(16, len(all_keys) * 0.4), 8))
    
    bars = plt.bar(range(len(all_keys)), counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for i, (key, count) in enumerate(zip(all_keys, counts)):
        if count > 0:
            plt.text(i, count, str(count), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xlabel("Cell Type Label", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title(f"{title}\nTotal: {total} cells", fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis labels
    plt.xticks(range(len(all_keys)), all_keys, rotation=90, ha='center', fontsize=8)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def write_csv_stats(output_path, dataset_name, label_counts, total_count, dict_keys):
    """Write CSV statistical file"""
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["Dataset Statistics", dataset_name])
        writer.writerow(["Total Images", total_count])
        writer.writerow([])
        
        # Detailed label statistics (show all dictionary keys in specified order)
        writer.writerow(["=== Detailed Label Statistics ==="])
        writer.writerow(["Label", "Count", "Percentage", "Order"])
        
        total_for_pct = total_count if total_count > 0 else 1
        
        for label in dict_keys:
            count = label_counts.get(label, 0)
            pct = count / total_for_pct * 100
            order = cell_dict.get(label, "")
            writer.writerow([label, count, f"{pct:.2f}%", order])
        
        # Summary statistics
        writer.writerow([])
        writer.writerow(["=== Summary ==="])
        writer.writerow(["Total Labels with Data", sum(1 for l in dict_keys if label_counts.get(l, 0) > 0)])
        writer.writerow(["Zero Count Labels", sum(1 for l in dict_keys if label_counts.get(l, 0) == 0)])

def process_single_directory(directory, output_dir):
    """
    Process single directory, generate statistical files
    Return statistical result dictionary (for global summary)
    """
    dir_path = Path(directory)
    dir_name = dir_path.name
    
    print(f"\n{'='*60}")
    print(f"📁 Processing directory: {dir_name}")
    print(f"   Path: {directory}")
    print(f"{'='*60}")
    
    # 1. Collect statistics
    label_counts, total_count, file_list = collect_image_stats(directory, recursive=True)
    
    if total_count == 0:
        print(f"⚠️ Warning: No valid images found in this directory!")
        return None
    
    print(f"✅ Found {total_count} images, {len(label_counts)} different labels")
    
    # 2. Generate bar chart
    bar_path = output_dir / f"{dir_name}_distribution_bar.png"
    create_bar_chart(label_counts, bar_path, f"Label Distribution - {dir_name}", 
                    DICT_KEYS, total_count)
    print(f"📊 Bar chart saved: {bar_path}")
    
    # 3. Generate CSV
    csv_path = output_dir / f"{dir_name}_statistics.csv"
    write_csv_stats(csv_path, dir_name, label_counts, total_count, DICT_KEYS)
    print(f"📄 CSV statistics saved: {csv_path}")
    
    # 4. Print summary
    nonzero_labels = [l for l in DICT_KEYS if label_counts.get(l, 0) > 0]
    zero_labels = [l for l in DICT_KEYS if label_counts.get(l, 0) == 0]
    
    # 修复：将长字符串分成多行，但确保每行都是完整的
    top_5 = sorted([(l, label_counts[l]) for l in nonzero_labels if l in label_counts], key=lambda x: -x[1])[:5]
    
    print(f"\n📊 Statistical Summary:")
    print(f"   Total Images: {total_count}")
    print(f"   Labels with data: {len(nonzero_labels)}/{len(DICT_KEYS)}")
    print(f"   Top 5 labels: {top_5}")
    
    if zero_labels:
        if len(zero_labels) > 10:
            print(f"   Zero count labels: {zero_labels[:10]}...")
        else:
            print(f"   Zero count labels: {zero_labels}")
    
    # 5. Return statistical results
    return {
        'directory': str(directory),
        'dir_name': dir_name,
        'label_counts': label_counts,
        'total_count': total_count
    }

def create_global_summary(all_stats, output_dir):
    """
    Create global summary statistics (merge all directories)
    """
    print(f"\n{'='*60}")
    print("📊 Generating global summary statistics...")
    print(f"{'='*60}")
    
    if not all_stats:
        print("⚠️ No valid data, skip global summary")
        return
    
    # 1. Merge all label statistics
    global_label_counts = collections.defaultdict(int)
    global_total = 0
    
    for stats in all_stats:
        for label, count in stats['label_counts'].items():
            global_label_counts[label] += count
        global_total += stats['total_count']
    
    # 2. Generate global bar chart
    bar_path = output_dir / "GLOBAL_distribution_bar.png"
    create_bar_chart(dict(global_label_counts), bar_path, "GLOBAL Label Distribution", 
                    DICT_KEYS, global_total)
    print(f"📊 Global bar chart saved: {bar_path}")
    
    # 3. Generate global CSV
    csv_path = output_dir / "GLOBAL_statistics.csv"
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["=== GLOBAL SUMMARY ==="])
        writer.writerow(["Total Datasets", len(all_stats)])
        writer.writerow(["Total Images", global_total])
        writer.writerow([])
        
        writer.writerow(["=== Detailed Label Statistics (All Datasets) ==="])
        writer.writerow(["Label", "Count", "Percentage", "Order"])
        
        total_for_pct = global_total if global_total > 0 else 1
        
        for label in DICT_KEYS:
            count = global_label_counts.get(label, 0)
            pct = count / total_for_pct * 100
            order = cell_dict.get(label, "")
            writer.writerow([label, count, f"{pct:.2f}%", order])
        
        writer.writerow([])
        writer.writerow(["=== Per-Dataset Summary ==="])
        writer.writerow(["Dataset", "Total Images", "Labels with Data"])
        for stats in all_stats:
            nonzero = sum(1 for l in DICT_KEYS if stats['label_counts'].get(l, 0) > 0)
            writer.writerow([
                stats['dir_name'],
                stats['total_count'],
                f"{nonzero}/{len(DICT_KEYS)}"
            ])
    
    print(f"📄 Global CSV statistics saved: {csv_path}")
    
    # 4. Print global summary
    nonzero_labels = [l for l in DICT_KEYS if global_label_counts.get(l, 0) > 0]
    
    # 修复：将长字符串分成多行，但确保每行都是完整的
    top_10 = sorted([(l, global_label_counts[l]) for l in nonzero_labels], key=lambda x: -x[1])[:10]
    
    print(f"\n📊 Global Statistical Summary:")
    print(f"   Number of Datasets: {len(all_stats)}")
    print(f"   Total Images: {global_total}")
    print(f"   Labels with data: {len(nonzero_labels)}/{len(DICT_KEYS)}")
    print(f"   Top 10 global labels: {top_10}")

# ==================== Main Program ====================

def main():
    print(f"🔍 Batch statistics script started")
    print(f"📁 Number of directories to process: {len(TARGET_DIRS)}")
    print(f"⚠️  Note: Labels not in dictionary will be categorized as '0'")
    print(f"📊 Dictionary keys ({len(DICT_KEYS)} labels): {DICT_KEYS}")
    
    # 1. Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_path}")
    
    # 2. Process directories one by one
    all_stats = []
    for i, directory in enumerate(TARGET_DIRS, 1):
        if not Path(directory).exists():
            print(f"\n⚠️ Warning: Directory does not exist, skip: {directory}")
            continue
        
        print(f"\n[{i}/{len(TARGET_DIRS)}] ", end="")
        stats = process_single_directory(directory, output_path)
        if stats:
            all_stats.append(stats)
    
    # 3. Generate global summary
    if all_stats:
        create_global_summary(all_stats, output_path)
    else:
        print("\n❌ No valid data, cannot generate global summary")
    
    # 4. Final summary
    print(f"\n{'='*60}")
    print("✨ Batch processing completed!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(all_stats)} directories")
    print(f"📂 All results saved to: {output_path}")
    print(f"\n📋 Generated file list:")
    for f in sorted(output_path.iterdir()):
        print(f"   - {f.name}")

if __name__ == "__main__":
    main()