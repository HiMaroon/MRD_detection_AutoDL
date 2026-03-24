import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

def rename_files_in_subfolders(root_dir: str, image_exts: List[str] = None) -> None:
    """
    批量重命名指定文件夹内所有一级子文件夹中的图片和JSON文件
    
    参数:
        root_dir: 根目录路径
        image_exts: 支持的图片扩展名列表（默认包含常见位图格式）
    """
    if image_exts is None:
        image_exts = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
    
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        raise ValueError(f"根目录不存在: {root_dir}")
    
    # 获取所有一级子文件夹
    subfolders = [p for p in root_path.iterdir() if p.is_dir()]
    if not subfolders:
        print(f"警告: 根目录 '{root_dir}' 下没有找到子文件夹")
        return
    
    print(f"开始处理 {len(subfolders)} 个子文件夹...\n")
    
    for subfolder in sorted(subfolders):
        process_subfolder(subfolder, image_exts)
    
    print("\n✓ 所有子文件夹处理完成！")


def process_subfolder(subfolder: Path, image_exts: List[str]) -> None:
    """处理单个子文件夹内的文件重命名"""
    print(f"处理子文件夹: {subfolder.name}")
    
    # 收集所有图片文件（按扩展名过滤）
    image_files = []
    for ext in image_exts:
        image_files.extend(subfolder.glob(f'*{ext}'))
        image_files.extend(subfolder.glob(f'*{ext.upper()}'))
    
    # 去重并排序（便于结果可重现）
    image_files = sorted(set(image_files))
    if not image_files:
        print(f"  ⚠ 跳过空文件夹: {subfolder.name}")
        return
    
    # 分组：有配对JSON的图片 和 无配对JSON的图片
    paired_images = []
    unpaired_images = []
    json_files = {}
    
    for img_path in image_files:
        json_path = img_path.with_suffix('.json')
        if json_path.exists():
            paired_images.append(img_path)
            json_files[img_path] = json_path
        else:
            unpaired_images.append(img_path)
    
    # 按规则排序：先处理有配对JSON的，再处理无配对的
    all_images = paired_images + unpaired_images
    
    # 生成新文件名映射
    mapping: Dict[str, str] = {}  # {旧文件名: 新文件名}
    processed_count = 0
    
    for img_path in all_images:
        processed_count += 1
        new_stem = f"{subfolder.name}-{processed_count}"
        
        # 确定新文件路径
        new_img_path = img_path.with_name(f"{new_stem}{img_path.suffix}")
        new_json_path = img_path.with_name(f"{new_stem}.json")
        
        # 记录映射关系
        mapping[img_path.name] = new_img_path.name
        if img_path in json_files:
            mapping[json_files[img_path].name] = new_json_path.name
        
        # 重命名JSON文件（如有）并更新内部imagePath
        if img_path in json_files:
            update_json_file(json_files[img_path], new_img_path.name, new_json_path)
        
        # 重命名图片文件（使用move避免覆盖风险）
        try:
            shutil.move(str(img_path), str(new_img_path))
        except Exception as e:
            print(f"  ✗ 重命名图片失败 {img_path.name} -> {new_img_path.name}: {e}")
            continue
    
    # 保存映射关系到txt文件
    save_mapping_file(subfolder, mapping, len(paired_images), len(unpaired_images))
    
    print(f"  ✓ 完成: {len(paired_images)} 对配对文件 + {len(unpaired_images)} 个孤立图片 → 共 {processed_count} 个文件")


def update_json_file(json_path: Path, new_image_name: str, new_json_path: Path) -> bool:
    """更新JSON文件中的imagePath字段并重命名文件"""
    try:
        # 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新imagePath
        old_path = data.get('imagePath', '')
        data['imagePath'] = new_image_name
        
        # 写回JSON（保留格式）
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 重命名JSON文件
        shutil.move(str(json_path), str(new_json_path))
        return True
    except Exception as e:
        print(f"  ✗ 处理JSON失败 {json_path.name}: {e}")
        return False


def save_mapping_file(subfolder: Path, mapping: Dict[str, str], paired_count: int, unpaired_count: int) -> None:
    """保存新旧文件名映射关系到txt文件"""
    mapping_file = subfolder / f"{subfolder.name}_rename_mapping.txt"
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write(f"子文件夹: {subfolder.name}\n")
        f.write(f"配对文件数量: {paired_count}\n")
        f.write(f"孤立图片数量: {unpaired_count}\n")
        f.write(f"总计重命名文件: {len(mapping)}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("旧文件名 -> 新文件名\n")
        f.write("="*60 + "\n\n")
        
        for old_name, new_name in sorted(mapping.items()):
            f.write(f"{old_name} -> {new_name}\n")
    
    print(f"  已保存映射记录: {mapping_file.name}")


if __name__ == "__main__":
    # ========== 配置区域 ==========
    ROOT_DIRECTORY = "/root/autodl-tmp/data/1"  # ⚠️ 请修改为您的实际路径
    
    # 可选：自定义支持的图片格式
    IMAGE_EXTENSIONS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
    # ============================
    
    try:
        rename_files_in_subfolders(ROOT_DIRECTORY, IMAGE_EXTENSIONS)
    except KeyboardInterrupt:
        print("\n⚠ 操作被用户中断")
    except Exception as e:
        print(f"\n✗ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()