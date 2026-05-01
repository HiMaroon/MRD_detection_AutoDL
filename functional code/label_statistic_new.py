# -*- coding: utf-8 -*-
import os
import json
import csv
import collections
from pathlib import Path

import matplotlib.pyplot as plt


# ==================== 配置区域 ====================

TARGET_DIRS = [
    "/root/autodl-tmp/data/MAIN_imgs_260323",
    "/root/autodl-tmp/data/BJH_imgs_260211",
    "/root/autodl-tmp/data/FXH_imgs_noALL_260318",
    "/root/autodl-tmp/data/TJMU_imgs_260416",
]

OUTPUT_DIR = "/root/autodl-tmp/data/json_label_stacked_statistics"

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


# ==================== JSON 读取统计 ====================

def collect_json_label_stats(directory):
    """
    仿照第一个代码：
    递归读取目录下所有 json 文件；
    只统计 shape_type == polygon 的标注；
    统计每个 label 的数量。
    """
    label_counts = collections.defaultdict(int)
    json_count = 0
    annotation_count = 0
    bad_jsons = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            json_path = os.path.join(root, filename)
            json_count += 1

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for shape in data.get("shapes", []):
                    if shape.get("shape_type") != "polygon":
                        continue

                    label = str(shape.get("label", "")).strip().upper()
                    if not label:
                        continue

                    if label not in cell_dict:
                        label = "0"

                    label_counts[label] += 1
                    annotation_count += 1

            except Exception as e:
                bad_jsons.append((json_path, str(e)))

    return {
        "directory": str(directory),
        "dir_name": Path(directory).name,
        "label_counts": dict(label_counts),
        "json_count": json_count,
        "annotation_count": annotation_count,
        "bad_jsons": bad_jsons,
    }


# ==================== 绘图函数 ====================

def get_folder_colors(n):
    cmap = plt.cm.tab20
    return [cmap(i % 20) for i in range(n)]


def create_single_bar_chart(stats, output_path):
    """
    单个目录普通柱状图。
    """
    label_counts = stats["label_counts"]
    counts = [label_counts.get(label, 0) for label in DICT_KEYS]
    total = sum(counts)

    plt.figure(figsize=(max(16, len(DICT_KEYS) * 0.45), 8))

    bars = plt.bar(
        range(len(DICT_KEYS)),
        counts,
        color="#5B8FF9",
        edgecolor="black",
        linewidth=0.5,
    )

    for i, count in enumerate(counts):
        if count > 0:
            plt.text(
                i, count, str(count),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )

    plt.xlabel("Cell Type Label", fontsize=12, fontweight="bold")
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.title(
        f"JSON Label Distribution - {stats['dir_name']}\n"
        f"Total annotations: {total}, JSON files: {stats['json_count']}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(range(len(DICT_KEYS)), DICT_KEYS, rotation=90, ha="center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_stacked_bar_chart(all_stats, output_path):
    """
    仿照第二个代码：
    每个 label 一个柱子；
    柱子内部按不同 TARGET_DIRS 文件夹堆叠；
    柱顶显示该 label 的总标注数。
    """
    if not all_stats:
        return

    dataset_names = [s["dir_name"] for s in all_stats]
    colors = get_folder_colors(len(all_stats))

    x = list(range(len(DICT_KEYS)))
    bottoms = [0] * len(DICT_KEYS)

    plt.figure(figsize=(max(18, len(DICT_KEYS) * 0.5), 9))

    for idx, stats in enumerate(all_stats):
        counts = [stats["label_counts"].get(label, 0) for label in DICT_KEYS]

        plt.bar(
            x,
            counts,
            bottom=bottoms,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
            label=dataset_names[idx],
        )

        bottoms = [bottoms[i] + counts[i] for i in range(len(DICT_KEYS))]

    total_counts = bottoms[:]
    global_total = sum(total_counts)

    for i, total in enumerate(total_counts):
        if total > 0:
            plt.text(
                i, total, str(total),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )

    plt.xlabel("Cell Type Label", fontsize=12, fontweight="bold")
    plt.ylabel("Annotation Count", fontsize=12, fontweight="bold")
    plt.title(
        f"GLOBAL JSON Label Distribution\n"
        f"Stacked by Folder, Total annotations: {global_total}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(range(len(DICT_KEYS)), DICT_KEYS, rotation=90, ha="center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="Folders", fontsize=9, title_fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# ==================== CSV 输出 ====================

def write_single_csv(stats, output_path):
    label_counts = stats["label_counts"]
    total = sum(label_counts.values())
    total_for_pct = total if total > 0 else 1

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Dataset", stats["dir_name"]])
        writer.writerow(["Directory", stats["directory"]])
        writer.writerow(["JSON Files", stats["json_count"]])
        writer.writerow(["Total Polygon Annotations", total])
        writer.writerow([])

        writer.writerow(["Label", "Count", "Percentage", "Order"])
        for label in DICT_KEYS:
            count = label_counts.get(label, 0)
            pct = count / total_for_pct * 100
            writer.writerow([label, count, f"{pct:.2f}%", cell_dict[label]])

        if stats["bad_jsons"]:
            writer.writerow([])
            writer.writerow(["=== Bad JSON Files ==="])
            writer.writerow(["Path", "Error"])
            for path, err in stats["bad_jsons"]:
                writer.writerow([path, err])


def write_global_csv(all_stats, output_path):
    global_counts = collections.defaultdict(int)

    for stats in all_stats:
        for label, count in stats["label_counts"].items():
            global_counts[label] += count

    global_total = sum(global_counts.values())
    total_for_pct = global_total if global_total > 0 else 1

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["=== GLOBAL SUMMARY ==="])
        writer.writerow(["Total Folders", len(all_stats)])
        writer.writerow(["Total JSON Files", sum(s["json_count"] for s in all_stats)])
        writer.writerow(["Total Polygon Annotations", global_total])
        writer.writerow([])

        writer.writerow(["=== Global Label Statistics ==="])
        writer.writerow(["Label", "Total Count", "Percentage", "Order"])
        for label in DICT_KEYS:
            count = global_counts.get(label, 0)
            pct = count / total_for_pct * 100
            writer.writerow([label, count, f"{pct:.2f}%", cell_dict[label]])

        writer.writerow([])
        writer.writerow(["=== Per Folder Label Statistics ==="])
        writer.writerow(["Folder", "Label", "Count"])

        for stats in all_stats:
            for label in DICT_KEYS:
                writer.writerow([
                    stats["dir_name"],
                    label,
                    stats["label_counts"].get(label, 0)
                ])


# ==================== 主程序 ====================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    print(f"📁 输出目录: {output_dir}")
    print(f"📊 统计 label 顺序: {DICT_KEYS}")

    for i, directory in enumerate(TARGET_DIRS, 1):
        directory = Path(directory)

        if not directory.exists():
            print(f"⚠️ 跳过不存在目录: {directory}")
            continue

        print(f"\n[{i}/{len(TARGET_DIRS)}] 正在统计: {directory}")

        stats = collect_json_label_stats(directory)

        if stats["annotation_count"] == 0:
            print(f"⚠️ 没有读取到 polygon 标注: {directory}")
            continue

        all_stats.append(stats)

        print(
            f"✅ {stats['dir_name']} | "
            f"JSON={stats['json_count']} | "
            f"Polygon annotations={stats['annotation_count']}"
        )

        single_png = output_dir / f"{stats['dir_name']}_json_label_bar.png"
        single_csv = output_dir / f"{stats['dir_name']}_json_label_statistics.csv"

        create_single_bar_chart(stats, single_png)
        write_single_csv(stats, single_csv)

        print(f"📊 单目录柱状图: {single_png}")
        print(f"📄 单目录CSV: {single_csv}")

    if all_stats:
        stacked_png = output_dir / "GLOBAL_json_label_stacked_bar.png"
        global_csv = output_dir / "GLOBAL_json_label_statistics.csv"

        create_stacked_bar_chart(all_stats, stacked_png)
        write_global_csv(all_stats, global_csv)

        print(f"\n✅ 全局堆叠柱状图: {stacked_png}")
        print(f"✅ 全局统计CSV: {global_csv}")
    else:
        print("\n❌ 没有有效 JSON 标注数据，未生成全局图。")


if __name__ == "__main__":
    main()