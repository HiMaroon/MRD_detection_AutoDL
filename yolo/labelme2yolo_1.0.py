import os
import json
import shutil
from pathlib import Path

# ===================== 配置参数 =====================

# 训练集路径列表
train_dirs = [
    "/root/autodl-tmp/data/MAIN_imgs_split_260323/Train",
]

# 验证集路径列表
val_dirs = [
    "/root/autodl-tmp/data/MAIN_imgs_split_260323/Val"
]

# 输出路径
output_base_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_dataset_260323"

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

# 类别映射
class_mapping = {
    "0": 0
}

# ===================== 处理函数 =====================
def process_json_files(json_files, image_output_dir, label_output_dir):
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
            # 这里的警告通常是因为 .ipynb_checkpoints 导致的，如果过滤了就不会再出现
            print(f"⚠️ 图像文件不存在: {image_path}")
            continue

        # 写入标签
        label_file = os.path.join(label_output_dir, Path(image_name).stem + ".txt")
        yolo_lines = []

        for shape in data.get("shapes", []):
            if shape["shape_type"] != "polygon":
                continue

            label = shape["label"]
            class_id = class_mapping.get(label, 0)

            points = shape["points"]
            normalized = []
            for x, y in points:
                normalized.append(f"{x / image_width:.6f}")
                normalized.append(f"{y / image_height:.6f}")

            line = f"{class_id} {' '.join(normalized)}"
            yolo_lines.append(line)

        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

# ===================== 收集并过滤 JSON 文件 (双重过滤) =====================

def get_filtered_jsons(directories):
    valid_files = []
    total_scanned = 0
    checkpoint_skipped = 0
    explicit_skipped = 0

    for d in directories:
        for json_file in Path(d).rglob("*.json"):
            total_scanned += 1
            
            # 1. 过滤 Jupyter 备份目录
            if ".ipynb_checkpoints" in str(json_file):
                checkpoint_skipped += 1
                continue
            
                
            valid_files.append(json_file)
            
    return valid_files, total_scanned, checkpoint_skipped, explicit_skipped

# 执行收集
train_json_files, t_total, t_cp, t_ex = get_filtered_jsons(train_dirs)
val_json_files, v_total, v_cp, v_ex = get_filtered_jsons(val_dirs)

print(f"📊 数据统计：")
print(f"  - 扫描到的总 JSON 数: {t_total + v_total}")
print(f"  - 因 Jupyter 备份跳过: {t_cp + v_cp}")
print(f"  - 因指定目录排除 (云紫梦): {t_ex + v_ex}")
print(f"  - 最终待处理 JSON 数: {len(train_json_files) + len(val_json_files)}")
print(f"📁 训练集 JSON 数量: {len(train_json_files)}")
print(f"📁 验证集 JSON 数量: {len(val_json_files)}")

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
yaml_path_formatted = output_base_dir.replace("\\", "/") # 兼容性处理

yaml_content = f"""# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
path: {yaml_path_formatted}
train: images/train
val: images/val
nc: {len(class_mapping)}
names:
"""
for label_name, class_id in class_mapping.items():
    yaml_content += f"  {class_id}: {label_name}\n"

with open(yaml_file_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"📄 配置文件已更新: {yaml_file_path}")