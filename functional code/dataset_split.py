import pandas as pd
import shutil
import os
from pathlib import Path

# ================= 配置区域 (请根据实际情况修改) =================
EXCEL_PATH = '/root/autodl-tmp/data/样本信息整理260323.xlsx'       # Excel 文件路径
SOURCE_ROOT = '/root/autodl-tmp/data/MAIN_imgs_260323'      # 存放所有原始样本子文件夹的目录
TARGET_ROOT = '/root/autodl-tmp/data/MAIN_imgs_split_260323'        # 希望整理到的目标根目录

# Excel 中的列名 (请确保与 Excel 表头完全一致)
COL_SAMPLE_NAME = '正式编号'
COL_SPLIT_TYPE = '划分'

# 划分类型的映射 (Excel 中的值 -> 目标文件夹名)
# 如果 Excel 里写的是 'Train'，这里会创建 'Train' 文件夹
SPLIT_MAPPING = {
    'Train': 'Train',
    'Valid': 'Val',
    'train': 'Train',   # 兼容小写
    'valid': 'Val',
    'val': 'Val'      # 兼容 val
}
# ===================================================================

def organize_dataset():
    # 1. 读取 Excel 文件
    try:
        df = pd.read_excel(EXCEL_PATH)
        print(f"✅ 成功读取 Excel 文件：{EXCEL_PATH}")
        print(f"📊 共读取到 {len(df)} 条样本记录\n")
    except Exception as e:
        print(f"❌ 读取 Excel 失败：{e}")
        return

    # 2. 检查源文件夹是否存在
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ 源文件夹不存在：{SOURCE_ROOT}")
        return

    # 3. 创建目标根目录
    Path(TARGET_ROOT).mkdir(parents=True, exist_ok=True)

    # 统计计数器
    count_success = 0
    count_failed = 0
    count_not_found = 0

    print("🚀 开始整理样本...\n")

    # 4. 遍历 Excel 每一行
    for index, row in df.iterrows():
        sample_name = str(row[COL_SAMPLE_NAME]).strip() # 去除可能的空格
        split_type_raw = str(row[COL_SPLIT_TYPE]).strip()
        
        # 标准化划分类型 (转为 Title 格式，如 train -> Train)
        split_type = SPLIT_MAPPING.get(split_type_raw, None)
        
        if not split_type:
            print(f"⚠️ 警告：第 {index+2} 行未知的划分类型 '{split_type_raw}'，已跳过。")
            count_failed += 1
            continue

        # 构建源路径和目标路径
        src_path = Path(SOURCE_ROOT) / sample_name
        dst_dir = Path(TARGET_ROOT) / split_type
        dst_path = dst_dir / sample_name

        # 检查源文件夹是否存在
        if not src_path.exists():
            print(f"❌ 未找到源文件夹：{src_path}")
            count_not_found += 1
            continue
        
        # 确保目标子文件夹 (Train/Valid) 存在
        dst_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 执行复制操作 (使用 copytree 复制整个文件夹)
            # dirs_exist_ok=True 表示如果目标已存在则覆盖 (Python 3.8+)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"✅ [{split_type}] {sample_name}")
            count_success += 1
        except Exception as e:
            print(f"❌ 复制失败 [{sample_name}]: {e}")
            count_failed += 1

    # 5. 输出总结报告
    print("\n" + "="*40)
    print("🎉 整理完成！")
    print(f"✅ 成功复制：{count_success} 个样本")
    print(f"⚠️ 源文件缺失：{count_not_found} 个样本 (Excel 有但文件夹没有)")
    print(f"❌ 其他失败：{count_failed} 个样本")
    print(f"📂 结果保存至：{os.path.abspath(TARGET_ROOT)}")
    print("="*40)

if __name__ == '__main__':
    organize_dataset()