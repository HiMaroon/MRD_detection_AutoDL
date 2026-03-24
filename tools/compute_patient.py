'''
#有可解释性
# tools/compute_patient.py
import pandas as pd
import os
import json
import collections
from pathlib import Path
from tqdm import tqdm

# ==================== 类别定义 ====================
cell_dict = {
    "V": 0, "0": 0, "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 0, "N2": 0, "N3": 0, "N4": 0, "N5": 0, "E": 0, "B": 0, "E1": 0, "B1": 0,
    "M0": 0, "M2": 0, "R2": 0, "R3": 0, "J2": 0, "J3": 0, "J4": 0, "P": 0, "P1": 0,
    "P2": 0, "P3": 0, "L": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0,
}

# ==================== 路径配置 (仅修改这里) ====================
# 1. 映射表路径 (aug_o_f)
mapping_xlsx      = "/root/autodl-tmp/projects/mwh/SingleCellProject/feature_o/aug_o_f/Val_mapping.xlsx"

# 2. 结果目录 (yolo_with_o_features)
results_dir       = "/root/autodl-tmp/results/yolo_with_o_features/"

# 自动查找最新的预测结果文件 (csv 或 xlsx)
files = [f for f in os.listdir(results_dir) if f.startswith("val_results") and (f.endswith('.csv') or f.endswith('.xlsx'))]
if not files:
    print(f"❌ 错误: 在 {results_dir} 没找到 val_results 文件")
    exit(1)
files.sort(reverse=True) # 取最新
results_xlsx = os.path.join(results_dir, files[0])

# 3. 其他固定路径
patient_type_xlsx = "/root/autodl-tmp/projects/mwh/SingleCellProject/yolo_o/训练样本分类.xlsx"
json_root_dir     = "/root/autodl-tmp/data/val_imgs_251127/"

# 4. 输出文件 (按您要求命名)
output_path       = "/root/autodl-tmp/results/yolo_with_o_features/patient_analysis_v_o_f.xlsx"

print(f"🚀 开始分析。预测结果文件: {results_xlsx}")

# ==================== 1. 数据加载与列名统一 ====================
if not Path(mapping_xlsx).exists():
    print(f"❌ 错误: 映射文件不存在 {mapping_xlsx}")
    exit(1)

# 读取映射表 (df1)
df1 = pd.read_excel(mapping_xlsx)
# 自动寻找并统一患者 ID 列名
map_p_col = [c for c in df1.columns if 'patient' in c.lower() or 'id' in c.lower()][0]
df1 = df1.rename(columns={map_p_col: 'Patient_ID'})

# 读取预测结果 (df2)
if results_xlsx.endswith('.csv'):
    df2 = pd.read_csv(results_xlsx)
else:
    df2 = pd.read_excel(results_xlsx)

img_col = [c for c in df2.columns if 'image' in c.lower() or 'file' in c.lower()][0]

# 读取患者类型表 (df3)
if os.path.exists(patient_type_xlsx):
    df3 = pd.read_excel(patient_type_xlsx, sheet_name='patient-id map valid')
    # 统一 df3 的患者 ID 列名为 'Patient_ID' 以便合并
    type_p_col = [c for c in df3.columns if 'patient' in c.lower() or 'id' in c.lower()][0]
    df3 = df3.rename(columns={type_p_col: 'Patient_ID'})
else:
    print("⚠️ 未找到患者类型表，将跳过类型合并")
    df3 = None

# ==================== 2. 统计原始 JSON 标注 ====================
print("正在统计原始图像中的细胞比例...")
patient_json_stats = collections.defaultdict(lambda: {'count_0': 0, 'count_1': 0})

json_files = list(Path(json_root_dir).rglob("*.json"))
for json_path in tqdm(json_files, desc="处理JSON"):
    p_id = json_path.parent.name  # 文件夹名即患者 ID
    try:
        data = json.load(json_path.open('r', encoding='utf-8'))
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            val = cell_dict.get(label, 0)
            if val == 1: patient_json_stats[p_id]['count_1'] += 1
            else: patient_json_stats[p_id]['count_0'] += 1
    except Exception as e:
        continue

json_df = pd.DataFrame.from_dict(patient_json_stats, orient='index').reset_index()
json_df.columns = ['Patient_ID', 'count_0', 'count_1']
json_df['json_total'] = json_df['count_0'] + json_df['count_1']
json_df['json_ratio_1'] = json_df['count_1'] / json_df['json_total'].clip(lower=1)

# ==================== 3. 汇总预测数据 ====================
print("合并预测结果与映射关系...")
df2['stem'] = df2[img_col].apply(lambda x: Path(x).stem)
# 寻找 df1 中的文件名列
cell_col = [c for c in df1.columns if 'cell' in c.lower() or 'filename' in c.lower()][0]

merged = pd.merge(df1[[cell_col, 'Patient_ID']], df2, left_on=cell_col, right_on='stem', how='inner')

if len(merged) == 0:
    print("❌ 错误: 合并后数据为 0，请检查文件名匹配逻辑。")
    exit(1)

print(f"成功合并 {len(merged)} 条细胞记录，开始按患者汇总...")
stats = merged.groupby('Patient_ID').agg(
    total_cells=('stem', 'count'),
    true_positive=('true_label', 'sum'),
    true_negative=('true_label', lambda x: (x == 0).sum()),
    pred_positive=('pred_label', 'sum'),
    pred_negative=('pred_label', lambda x: (x == 0).sum()),
    true_labels=('true_label', list),
    pred_labels=('pred_label', list)
).reset_index()

stats['actual_ratio'] = stats['true_positive'] / stats['total_cells']
stats['predicted_ratio'] = stats['pred_positive'] / stats['total_cells']

# ==================== 4. 最终合并 ====================
# 合并患者类型
if df3 is not None:
    result = pd.merge(stats, df3[['Patient_ID', 'type']], on='Patient_ID', how='left')
else:
    result = stats
    result['type'] = 'Unknown'

# 合并 JSON 统计
result = pd.merge(result, json_df, on='Patient_ID', how='left')

# 计算准确率
result['correct_predictions'] = result.apply(
    lambda row: sum(t == p for t, p in zip(row['true_labels'], row['pred_labels'])), axis=1
)
result['accuracy'] = result['correct_predictions'] / result['total_cells']

# 整理列顺序
final_cols = [
    'Patient_ID', 'type', 'total_cells', 'true_positive', 'true_negative', 
    'actual_ratio', 'pred_positive', 'pred_negative', 'predicted_ratio',
    'count_0', 'count_1', 'json_total', 'json_ratio_1', 'accuracy'
]
# 确保列都存在再筛选
final_cols = [c for c in final_cols if c in result.columns]
result = result[final_cols]

# ==================== 5. 保存结果 ====================
result.to_excel(output_path, index=False)
print(f"\n✅ 分析完成！结果已保存至: {output_path}")
print(f"共分析了 {len(result)} 位患者。")
print(result[['Patient_ID', 'type', 'actual_ratio', 'predicted_ratio', 'accuracy']].head())
'''



# tools/compute_patient.py - 无可解释性
import pandas as pd
import os
import json
import collections
from pathlib import Path
from tqdm import tqdm

# ==================== 类别定义 ====================
cell_dict = {
    "V": 0, "0": 0, "N": 1, "N1": 1, "M": 1, "M1": 1, "R": 1, "R1": 1, "J": 1, "J1": 1,
    "N0": 0, "N2": 0, "N3": 0, "N4": 0, "N5": 0, "E": 0, "B": 0, "E1": 0, "B1": 0,
    "M0": 0, "M2": 0, "R2": 0, "R3": 0, "J2": 0, "J3": 0, "J4": 0, "P": 0, "P1": 0,
    "P2": 0, "P3": 0, "L": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0,
}

# ==================== 路径配置 ====================
# 请根据实际生成的文件名微调 results_xlsx 的时间戳
mapping_xlsx      = "/root/autodl-tmp/projects/mwh/SingleCellProject/cellseg_m/Val_mapping.xlsx"

results_xlsx      = "/root/autodl-tmp/results/cellpose_with_modified_data/val_results_20260108-113716.xlsx" #/root/autodl-tmp/results/yolo_with_original_data_pretrained/val_results_20251225-025957.xlsx   /root/autodl-tmp/results/yolo_with_modified_data/val_results_20251224-152112.xlsx /root/autodl-tmp/results/yolo_with_original_data/val_results_20251224-150009.xlsx  /root/autodl-tmp/results/yolo_with_modified_data_pretrained/val_results_20260107-211657.xlsx  /root/autodl-tmp/results/cellpose_with_original_data/val_results_20260108-100731.xlsx

patient_type_xlsx = "/root/autodl-tmp/projects/mwh/SingleCellProject/yolo_o/训练样本分类.xlsx"
json_root_dir     = "/root/autodl-tmp/data/val_imgs_251127/"
output_path       = "/root/autodl-tmp/results/cellpose_with_modified_data/patient_analysis_v_m.xlsx"

print(f"🚀 开始分析。预测结果文件: {results_xlsx}")

# ==================== 1. 数据加载与列名统一 ====================
if not Path(mapping_xlsx).exists():
    print(f"❌ 错误: 映射文件不存在 {mapping_xlsx}")
    exit(1)

# 读取映射表 (df1)
df1 = pd.read_excel(mapping_xlsx)
# 自动寻找并统一患者 ID 列名
map_p_col = [c for c in df1.columns if 'patient' in c.lower() or 'id' in c.lower()][0]
df1 = df1.rename(columns={map_p_col: 'Patient_ID'})

# 读取预测结果 (df2)
df2 = pd.read_excel(results_xlsx)
img_col = [c for c in df2.columns if 'image' in c.lower() or 'file' in c.lower()][0]

# 读取患者类型表 (df3)
df3 = pd.read_excel(patient_type_xlsx, sheet_name='patient-id map valid')
# 统一 df3 的患者 ID 列名为 'Patient_ID' 以便合并
type_p_col = [c for c in df3.columns if 'patient' in c.lower() or 'id' in c.lower()][0]
df3 = df3.rename(columns={type_p_col: 'Patient_ID'})

# ==================== 2. 统计原始 JSON 标注 ====================
print("正在统计原始图像中的细胞比例...")
patient_json_stats = collections.defaultdict(lambda: {'count_0': 0, 'count_1': 0})

json_files = list(Path(json_root_dir).rglob("*.json"))
for json_path in tqdm(json_files, desc="处理JSON"):
    p_id = json_path.parent.name  # 文件夹名即患者 ID
    try:
        data = json.load(json_path.open('r', encoding='utf-8'))
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            val = cell_dict.get(label, 0)
            if val == 1: patient_json_stats[p_id]['count_1'] += 1
            else: patient_json_stats[p_id]['count_0'] += 1
    except Exception as e:
        continue

json_df = pd.DataFrame.from_dict(patient_json_stats, orient='index').reset_index()
json_df.columns = ['Patient_ID', 'count_0', 'count_1']
json_df['json_total'] = json_df['count_0'] + json_df['count_1']
json_df['json_ratio_1'] = json_df['count_1'] / json_df['json_total'].clip(lower=1)

# ==================== 3. 汇总预测数据 ====================
print("合并预测结果与映射关系...")
df2['stem'] = df2[img_col].apply(lambda x: Path(x).stem)
# 寻找 df1 中的文件名列
cell_col = [c for c in df1.columns if 'cell' in c.lower() or 'filename' in c.lower()][0]

merged = pd.merge(df1[[cell_col, 'Patient_ID']], df2, left_on=cell_col, right_on='stem', how='inner')

if len(merged) == 0:
    print("❌ 错误: 合并后数据为 0，请检查文件名匹配逻辑。")
    exit(1)

print(f"成功合并 {len(merged)} 条细胞记录，开始按患者汇总...")
stats = merged.groupby('Patient_ID').agg(
    total_cells=('stem', 'count'),
    true_positive=('true_label', 'sum'),
    true_negative=('true_label', lambda x: (x == 0).sum()),
    pred_positive=('pred_label', 'sum'),
    pred_negative=('pred_label', lambda x: (x == 0).sum()),
    true_labels=('true_label', list),
    pred_labels=('pred_label', list)
).reset_index()

stats['actual_ratio'] = stats['true_positive'] / stats['total_cells']
stats['predicted_ratio'] = stats['pred_positive'] / stats['total_cells']

# ==================== 4. 最终合并 ====================
# 合并患者类型
result = pd.merge(stats, df3[['Patient_ID', 'type']], on='Patient_ID', how='left')
# 合并 JSON 统计
result = pd.merge(result, json_df, on='Patient_ID', how='left')

# 计算准确率
result['correct_predictions'] = result.apply(
    lambda row: sum(t == p for t, p in zip(row['true_labels'], row['pred_labels'])), axis=1
)
result['accuracy'] = result['correct_predictions'] / result['total_cells']

# 整理列顺序
final_cols = [
    'Patient_ID', 'type', 'total_cells', 'true_positive', 'true_negative', 
    'actual_ratio', 'pred_positive', 'pred_negative', 'predicted_ratio',
    'count_0', 'count_1', 'json_total', 'json_ratio_1', 'accuracy'
]
result = result[final_cols]

# ==================== 5. 保存结果 ====================
result.to_excel(output_path, index=False)
print(f"\n✅ 分析完成！结果已保存至: {output_path}")
print(f"共分析了 {len(result)} 位患者。")
print(result[['Patient_ID', 'type', 'actual_ratio', 'predicted_ratio', 'accuracy']].head())
