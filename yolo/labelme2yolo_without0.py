import os
import json
import shutil
from pathlib import Path

# ===================== 配置参数 =====================

# 训练集路径列表
train_dirs = [
    "/root/autodl-tmp/data/MAIN_imgs_split_260312/Train",
]

# 验证集路径列表
val_dirs = [
    "/root/autodl-tmp/data/MAIN_imgs_split_260312/Val"
]

# 输出路径
output_base_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_dataset_without0"

output_train_image_dir = os.path.join(output_base_dir, "images", "train")
output_val_image_dir = os.path.join(output_base_dir, "images", "val")
output_train_label_dir = os.path.join(output_base_dir, "labels", "train")
output_val_label_dir = os.path.join(output_base_dir, "labels", "val")
source_record_file = os.path.join(output_base_dir, "source_dirs.txt")

# 创建输出目录
os.makedirs(output_train_image_dir, exist_ok=True)
os.makedirs(output_val_image_dir, exist_ok=True)
os.makedirs(output_train_label_dir, exist_ok=True)
os.makedirs(output_val_label_dir, exist_ok=True)

# ===================== 类别映射配置 =====================
# 只保留这些类别，其余全部过滤
cell_dict_big = {
    # "V": 0, "0": 0,
    "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2,
    "M0": 2, "M2": 2, "R2": 2, "R3": 2,
    "J2": 2, "J3": 2, "J4": 2,
    "P": 2, "P1": 2, "P2": 2, "P3": 2,
    "L": 2, "L1": 2, "L2": 2, "L3": 2, "L4": 2
}

# 构建 class_mapping：cell_dict_big 的 key -> 0（YOLO 类别 ID）
class_mapping = {label: 0 for label in cell_dict_big.keys()}

# ===================== 处理函数 =====================
def process_json_files(json_files, image_output_dir, label_output_dir):
    skipped_labels = set()  # 记录被跳过的标签
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ JSON 文件加载失败: {json_file}，错误：{e}")
            continue

        image_name = data.get("imagePath")
        image_path = json_file.parent / image_name
        image_height = data.get("imageHeight")
        image_width = data.get("imageWidth")

        if not all([image_name, image_height, image_width]):
            print(f"⚠️ JSON 文件信息不全: {json_file}")
            continue

        # 复制图像
        dst_img_path = os.path.join(image_output_dir, image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, dst_img_path)
        else:
            print(f"⚠️ 图像文件不存在: {image_path}")
            continue

        # 写入标签
        label_file = os.path.join(label_output_dir, Path(image_name).stem + ".txt")
        yolo_lines = []

        for shape in data.get("shapes", []):
            if shape["shape_type"] != "polygon":
                continue

            label = shape["label"]
            
            # 只保留 cell_dict_big 中定义的类别
            if label not in cell_dict_big:
                skipped_labels.add(label)
                continue
            
            # 统一映射为类别 0
            class_id = 0

            points = shape["points"]
            normalized = []
            for x, y in points:
                normalized.append(f"{x / image_width:.6f}")
                normalized.append(f"{y / image_height:.6f}")

            line = f"{class_id} {' '.join(normalized)}"
            yolo_lines.append(line)

        # 只有包含有效标签时才写入标签文件
        if yolo_lines:
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
        else:
            # 可选：如果没有有效标签，可以删除对应的图像
            print(f"⚠️ 图像 {image_name} 没有有效标签，跳过")
            os.remove(dst_img_path)
    
    # 打印被跳过的标签统计
    if skipped_labels:
        print(f"   被过滤的标签: {skipped_labels}")

# ===================== 收集并过滤 JSON 文件 (双重过滤) =====================

def get_filtered_jsons(directories):
    valid_files = []
    total_scanned = 0
    checkpoint_skipped = 0

    for d in directories:
        for json_file in Path(d).rglob("*.json"):
            total_scanned += 1
            
            # 过滤 Jupyter 备份目录
            if ".ipynb_checkpoints" in str(json_file):
                checkpoint_skipped += 1
                continue
            
            valid_files.append(json_file)
            
    return valid_files, total_scanned, checkpoint_skipped

# 执行收集
train_json_files, t_total, t_cp = get_filtered_jsons(train_dirs)
val_json_files, v_total, v_cp = get_filtered_jsons(val_dirs)

print(f"📊 数据统计：")
print(f"  - 扫描到的总 JSON 数: {t_total + v_total}")
print(f"  - 因 Jupyter 备份跳过: {t_cp + v_cp}")
print(f"  - 最终待处理 JSON 数: {len(train_json_files) + len(val_json_files)}")
print(f"📁 训练集 JSON 数量: {len(train_json_files)}")
print(f"📁 验证集 JSON 数量: {len(val_json_files)}")
print(f"🎯 保留的类别: {list(cell_dict_big.keys())}")

# ===================== 生成来源记录文件 =====================
with open(source_record_file, 'w', encoding='utf-8') as f:
    f.write("# ========== 训练集来源路径 ==========\n")
    for d in train_dirs: f.write(f"{d}\n")
    f.write("\n# ========== 验证集来源路径 ==========\n")
    for d in val_dirs: f.write(f"{d}\n")

# ===================== 执行处理 =====================
print("📦 正在处理训练集...")
process_json_files(train_json_files, output_train_image_dir, output_train_label_dir)

print("📦 正在处理验证集...")
process_json_files(val_json_files, output_val_image_dir, output_val_label_dir)

print("✅ 数据转换完成。")

# ===================== 生成 dataset.yaml =====================
yaml_file_path = os.path.join(output_base_dir, "dataset.yaml")
yaml_path_formatted = output_base_dir.replace("\\", "/")

yaml_content = f"""# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
path: {yaml_path_formatted}
train: images/train
val: images/val
nc: 1
names:
  0: cell
"""

with open(yaml_file_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"📄 配置文件已更新: {yaml_file_path}")