# -*- coding: utf-8 -*-
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import csv


"""
Batch statistics for image category distribution.

功能：
1. 从图片文件名最后一个下划线后的字段读取 label：
   xxx_xxx_LABEL.png -> LABEL

2. train 和 val 合并为 MAIN

3. 输出：
   - 每个 group 的普通柱状图
   - 每个 group 的 CSV
   - 全局堆叠柱状图
   - 全局统计 CSV

4. 全局堆叠柱状图：
   - 每个 label 一个柱子
   - 柱子内部按 MAIN / test_FXH_noALL / test_BJH / test_TJMU 分层
   - 柱顶显示该 label 的总数
"""


# ==================== Configuration Area ====================

TARGET_GROUPS = {
    "MAIN": [
        "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train",
        "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val",
    ],
    "test_FXH_noALL": [
        "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL",
    ],
    "test_BJH": [
        "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH",
    ],
    "test_TJMU": [
        "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU",
    ],
}

OUTPUT_DIR = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/statistics_all_MAIN_grouped"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

cell_dict = {
    "N0": 1, "N": 2, "N1": 3, "N2": 4, "N3": 5, "N4": 6, "N5": 7,
    "E": 8, "E1": 9,
    "B": 10, "B1": 11,
    "M0": 12, "M": 13, "M1": 14, "M2": 15,
    "R": 16, "R1": 17, "R2": 18, "R3": 19,
    "J": 20, "J1": 21, "J2": 22, "J3": 23, "J4": 24,
    "L": 25, "L1": 26, "L2": 27, "L3": 28, "L4": 29,
    "P": 30, "P1": 31, "P2": 32, "P3": 33,
    "A": 34, "F": 35, "V": 36, "0": 37,
}

DICT_KEYS = sorted(cell_dict.keys(), key=lambda x: cell_dict[x])


# ==================== Core Functions ====================

def extract_label_from_filename(filename):
    """
    Extract label from filename.
    Example:
        PKUPH-106-10_000_P2.png -> P2
    """
    name_no_ext = Path(filename).stem
    parts = name_no_ext.rsplit("_", 1)
    if len(parts) >= 2:
        return parts[-1].strip().upper()
    return None


def collect_image_stats(directory, recursive=True):
    """
    Traverse directory to collect image label statistics.
    """
    label_counts = collections.defaultdict(int)
    file_list = []
    dir_path = Path(directory)

    if recursive:
        files = dir_path.rglob("*")
    else:
        files = dir_path.iterdir()

    for file_path in files:
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = extract_label_from_filename(file_path.name)

        if label:
            if label not in cell_dict:
                label = "0"

            label_counts[label] += 1
            file_list.append((str(file_path), label))

    return dict(label_counts), sum(label_counts.values()), file_list


def get_single_bar_color():
    return "#5B8FF9"


def get_folder_colors(n):
    cmap = plt.cm.tab20
    return [cmap(i % 20) for i in range(n)]


def create_bar_chart(label_counts, output_path, title, dict_keys, total_count=None):
    """
    单个 group 的普通柱状图：
    - 所有柱子同一种颜色
    - 每个柱子顶部显示数字
    """
    counts = [label_counts.get(key, 0) for key in dict_keys]
    total = total_count or sum(counts)

    plt.figure(figsize=(max(16, len(dict_keys) * 0.4), 8))

    plt.bar(
        range(len(dict_keys)),
        counts,
        color=get_single_bar_color(),
        edgecolor="black",
        linewidth=0.5,
    )

    for i, count in enumerate(counts):
        if count > 0:
            plt.text(
                i,
                count,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.xlabel("Cell Type Label", fontsize=12, fontweight="bold")
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.title(
        f"{title}\nTotal: {total} cells",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(range(len(dict_keys)), dict_keys, rotation=90, ha="center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_stacked_bar_chart(all_stats, output_path, title, dict_keys):
    """
    全局堆叠柱状图：
    - 每个 label 一个柱子
    - 柱子内部按不同 group 堆叠
    - 每层不同颜色
    - 柱顶显示总数
    """
    if not all_stats:
        return

    dataset_names = [stats["dir_name"] for stats in all_stats]
    folder_colors = get_folder_colors(len(all_stats))

    x = list(range(len(dict_keys)))
    bottoms = [0] * len(dict_keys)

    plt.figure(figsize=(max(18, len(dict_keys) * 0.45), 9))

    for idx, stats in enumerate(all_stats):
        counts = [stats["label_counts"].get(label, 0) for label in dict_keys]

        plt.bar(
            x,
            counts,
            bottom=bottoms,
            color=folder_colors[idx],
            edgecolor="black",
            linewidth=0.4,
            label=dataset_names[idx],
        )

        bottoms = [bottoms[i] + counts[i] for i in range(len(dict_keys))]

    total_counts = bottoms[:]
    global_total = sum(total_counts)

    for i, total in enumerate(total_counts):
        if total > 0:
            plt.text(
                i,
                total,
                str(total),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.xlabel("Cell Type Label", fontsize=12, fontweight="bold")
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.title(
        f"{title}\nTotal: {global_total} cells",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(range(len(dict_keys)), dict_keys, rotation=90, ha="center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="Groups", fontsize=9, title_fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def write_csv_stats(output_path, dataset_name, label_counts, total_count, dict_keys):
    """
    Write CSV statistical file for a single group.
    """
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Dataset Statistics", dataset_name])
        writer.writerow(["Total Images", total_count])
        writer.writerow([])

        writer.writerow(["=== Detailed Label Statistics ==="])
        writer.writerow(["Label", "Count", "Percentage", "Order"])

        total_for_pct = total_count if total_count > 0 else 1

        for label in dict_keys:
            count = label_counts.get(label, 0)
            pct = count / total_for_pct * 100
            order = cell_dict.get(label, "")
            writer.writerow([label, count, f"{pct:.2f}%", order])

        writer.writerow([])
        writer.writerow(["=== Summary ==="])
        writer.writerow([
            "Total Labels with Data",
            sum(1 for l in dict_keys if label_counts.get(l, 0) > 0),
        ])
        writer.writerow([
            "Zero Count Labels",
            sum(1 for l in dict_keys if label_counts.get(l, 0) == 0),
        ])


def process_group(group_name, directories, output_dir):
    """
    处理一个 group。

    例如：
    MAIN = train + val

    返回给全局堆叠图使用的统计结果。
    """
    group_label_counts = collections.defaultdict(int)
    group_total_count = 0
    group_file_list = []

    print(f"\n{'=' * 60}")
    print(f"📁 Processing group: {group_name}")
    print(f"{'=' * 60}")

    for directory in directories:
        directory = Path(directory)

        if not directory.exists():
            print(f"⚠️ Warning: Directory does not exist, skip: {directory}")
            continue

        label_counts, total_count, file_list = collect_image_stats(directory, recursive=True)

        print(f"   - {directory.name}: {total_count} images")

        for label, count in label_counts.items():
            group_label_counts[label] += count

        group_total_count += total_count
        group_file_list.extend(file_list)

    if group_total_count == 0:
        print(f"⚠️ Warning: group {group_name} has no valid images")
        return None

    group_label_counts = dict(group_label_counts)

    bar_path = output_dir / f"{group_name}_distribution_bar.png"
    create_bar_chart(
        group_label_counts,
        bar_path,
        f"Label Distribution - {group_name}",
        DICT_KEYS,
        group_total_count,
    )

    csv_path = output_dir / f"{group_name}_statistics.csv"
    write_csv_stats(
        csv_path,
        group_name,
        group_label_counts,
        group_total_count,
        DICT_KEYS,
    )

    nonzero_labels = [l for l in DICT_KEYS if group_label_counts.get(l, 0) > 0]
    zero_labels = [l for l in DICT_KEYS if group_label_counts.get(l, 0) == 0]
    top_5 = sorted(
        [(l, group_label_counts[l]) for l in nonzero_labels if l in group_label_counts],
        key=lambda x: -x[1],
    )[:5]

    print(f"\n📊 Statistical Summary for {group_name}:")
    print(f"   Total Images: {group_total_count}")
    print(f"   Labels with data: {len(nonzero_labels)}/{len(DICT_KEYS)}")
    print(f"   Top 5 labels: {top_5}")

    if zero_labels:
        if len(zero_labels) > 10:
            print(f"   Zero count labels: {zero_labels[:10]}...")
        else:
            print(f"   Zero count labels: {zero_labels}")

    print(f"📊 Bar chart saved: {bar_path}")
    print(f"📄 CSV statistics saved: {csv_path}")

    return {
        "directory": ";".join([str(d) for d in directories]),
        "dir_name": group_name,
        "label_counts": group_label_counts,
        "total_count": group_total_count,
    }


def create_global_summary(all_stats, output_dir):
    """
    Create global summary statistics.
    """
    print(f"\n{'=' * 60}")
    print("📊 Generating global summary statistics...")
    print(f"{'=' * 60}")

    if not all_stats:
        print("⚠️ No valid data, skip global summary")
        return

    global_label_counts = collections.defaultdict(int)
    global_total = 0

    for stats in all_stats:
        for label, count in stats["label_counts"].items():
            global_label_counts[label] += count
        global_total += stats["total_count"]

    stacked_bar_path = output_dir / "GLOBAL_distribution_stacked_bar.png"
    create_stacked_bar_chart(
        all_stats,
        stacked_bar_path,
        "GLOBAL Label Distribution: MAIN vs External Test Sets",
        DICT_KEYS,
    )
    print(f"📊 Global stacked bar chart saved: {stacked_bar_path}")

    csv_path = output_dir / "GLOBAL_statistics.csv"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["=== GLOBAL SUMMARY ==="])
        writer.writerow(["Total Groups", len(all_stats)])
        writer.writerow(["Total Images", global_total])
        writer.writerow([])

        writer.writerow(["=== Detailed Label Statistics All Groups ==="])
        writer.writerow(["Label", "Count", "Percentage", "Order"])

        total_for_pct = global_total if global_total > 0 else 1

        for label in DICT_KEYS:
            count = global_label_counts.get(label, 0)
            pct = count / total_for_pct * 100
            order = cell_dict.get(label, "")
            writer.writerow([label, count, f"{pct:.2f}%", order])

        writer.writerow([])
        writer.writerow(["=== Per-Group Summary ==="])
        writer.writerow(["Group", "Total Images", "Labels with Data"])

        for stats in all_stats:
            nonzero = sum(1 for l in DICT_KEYS if stats["label_counts"].get(l, 0) > 0)
            writer.writerow([
                stats["dir_name"],
                stats["total_count"],
                f"{nonzero}/{len(DICT_KEYS)}",
            ])

        writer.writerow([])
        writer.writerow(["=== Per-Group Per-Label Count Matrix ==="])

        header = ["Label"] + [stats["dir_name"] for stats in all_stats] + ["Total"]
        writer.writerow(header)

        for label in DICT_KEYS:
            row = [label]
            label_total = 0

            for stats in all_stats:
                count = stats["label_counts"].get(label, 0)
                row.append(count)
                label_total += count

            row.append(label_total)
            writer.writerow(row)

    print(f"📄 Global CSV statistics saved: {csv_path}")

    nonzero_labels = [l for l in DICT_KEYS if global_label_counts.get(l, 0) > 0]
    top_10 = sorted(
        [(l, global_label_counts[l]) for l in nonzero_labels],
        key=lambda x: -x[1],
    )[:10]

    print(f"\n📊 Global Statistical Summary:")
    print(f"   Number of Groups: {len(all_stats)}")
    print(f"   Total Images: {global_total}")
    print(f"   Labels with data: {len(nonzero_labels)}/{len(DICT_KEYS)}")
    print(f"   Top 10 global labels: {top_10}")


# ==================== Main Program ====================

def main():
    print("🔍 Batch statistics script started")
    print(f"📁 Number of groups to process: {len(TARGET_GROUPS)}")
    print("⚠️  Note: Labels not in dictionary will be categorized as '0'")
    print(f"📊 Dictionary keys ({len(DICT_KEYS)} labels): {DICT_KEYS}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_path}")

    all_stats = []

    for i, (group_name, directories) in enumerate(TARGET_GROUPS.items(), 1):
        print(f"\n[{i}/{len(TARGET_GROUPS)}] ", end="")
        stats = process_group(group_name, directories, output_path)

        if stats:
            all_stats.append(stats)

    if all_stats:
        create_global_summary(all_stats, output_path)
    else:
        print("\n❌ No valid data, cannot generate global summary")

    print(f"\n{'=' * 60}")
    print("✨ Batch processing completed!")
    print(f"{'=' * 60}")
    print(f"✅ Successfully processed: {len(all_stats)} groups")
    print(f"📂 All results saved to: {output_path}")

    print("\n📋 Generated file list:")
    for f in sorted(output_path.iterdir()):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()