#可解释性版 tools/patient_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
import os
import matplotlib.font_manager as fm

# ==================== 配置 ====================
root_path = "/root/autodl-tmp/results/yolo_with_original_data_pretrained/"     #"/root/autodl-tmp/results/yolo_with_o_features/"
input_file = "patient_analysis_v_o.xlsx"
output_file = "patient_ratios_comparison_v_o.png"

# 这里的名字必须和您上传的文件名大小写完全一致
FONT_NAME = "MSYH.TTC" 

def get_full_path(filename):
    return os.path.join(root_path, filename)

def set_chinese_font():
    """尝试加载中文字体，解决乱码问题 (修复版)"""
    # 1. 寻找字体文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, FONT_NAME),           # 脚本同级目录
        os.path.join(os.getcwd(), FONT_NAME),           # 运行时当前目录
        os.path.join(root_path, FONT_NAME),             # 结果输出目录
        f"/root/autodl-tmp/projects/mwh/SingleCellProject/tools/{FONT_NAME}" 
    ]
    
    selected_font = None
    for path in candidates:
        if os.path.exists(path):
            selected_font = path
            break
            
    if selected_font:
        print(f"✅ 找到字体文件: {selected_font}")
        try:
            # === 核心修复：显式注册字体文件 ===
            fm.fontManager.addfont(selected_font)
            
            # 获取字体的正式名称 (如 'Microsoft YaHei UI')
            font_prop = fm.FontProperties(fname=selected_font)
            font_name = font_prop.get_name()
            
            # 设置为全局默认字体
            plt.rcParams['font.family'] = font_name
            print(f"   已注册并设置为默认字体: {font_name}")
        except Exception as e:
            print(f"⚠️ 字体注册异常: {e}")
    else:
        print(f"⚠️ 未找到 {FONT_NAME}，尝试使用系统回退字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    
    plt.rcParams['axes.unicode_minus'] = False

def main():
    set_chinese_font()
    
    file_path = get_full_path(input_file)
    if not os.path.exists(file_path):
        print(f"❌ 找不到输入文件: {file_path}")
        return

    print(f"📖 读取数据: {file_path}")
    result_df = pd.read_excel(file_path)

    # 筛选 AML 和 HD
    target_types = ['AML', 'HD']
    if result_df['type'].isin(target_types).any():
        plot_df = result_df[result_df['type'].isin(target_types)].copy()
    else:
        plot_df = result_df.copy()

    # === 计算最佳阈值 ===
    best_thresh = 0
    if 'AML' in plot_df['type'].unique() and 'HD' in plot_df['type'].unique():
        try:
            y_true = (plot_df['type'] == 'AML').astype(int)
            y_score = plot_df['predicted_ratio']
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            if len(thresholds) > 0:
                J = tpr - fpr
                best_thresh = thresholds[np.argmax(J)]
        except:
            pass

    # === 排序逻辑：升序 (HD在左，AML在右) ===
    plot_df = plot_df.sort_values('predicted_ratio', ascending=True)
    
    # 颜色映射
    plot_df['color'] = plot_df['type'].map({'AML': 'red', 'HD': 'blue'}).fillna('black')

    # === 绘图 ===
    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    n_patients = len(plot_df)
    x = np.arange(n_patients)
    width = 0.3

    # 绘制柱状图
    rects1 = ax.bar(x - width, plot_df['actual_ratio'], width, color='lightcoral', edgecolor='red', label='实际比例')
    rects2 = ax.bar(x, plot_df['predicted_ratio'], width, color='lightblue', edgecolor='blue', label='预测比例')
    
    has_json = 'json_ratio_1' in plot_df.columns
    if has_json:
        rects3 = ax.bar(x + width, plot_df['json_ratio_1'], width, color='lightgreen', edgecolor='green', label='JSON标注比例')

    # 底部标注类型
    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(x[i], -0.05, str(row['type']), ha='center', va='top', rotation=90, color=row['color'], fontweight='bold')

    # 数值标签
    def add_value_labels(rects, offset=0.02):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + offset, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    add_value_labels(rects1)
    add_value_labels(rects2)
    if has_json:
        add_value_labels(rects3)

    # 图例
    legend_elements = [
        Line2D([0], [0], color='lightcoral', lw=4, label='实际原始细胞比例'),
        Line2D([0], [0], color='lightblue', lw=4, label='预测原始细胞比例'),
    ]
    if has_json:
        legend_elements.append(Line2D([0], [0], color='lightgreen', lw=4, label='JSON标注原始细胞比例'))

    ax.set_ylabel('原始细胞比例', fontsize=12)
    ax.set_xlabel('患者 (按预测比例排序: HD → AML)', fontsize=12, labelpad=50)
    ax.set_title('各患者原始细胞比例对比 (实际 vs 预测)', fontsize=14, pad=20)
    
    # X轴刻度
    ax.set_xticks(x)
    labels = [str(pid)[:15] for pid in plot_df['Patient_ID']]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = get_full_path(output_file)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ 图形已保存: {save_path}")
    if best_thresh > 0:
        print(f"📊 最佳区分阈值: {best_thresh:.4f}")

if __name__ == "__main__":
    main()


'''
#原版
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
import os

root_path = "/root/autodl-tmp/projects/mwh/SingleCellProject/outputs/results"

def get_full_path(filename):
    return os.path.join(root_path, filename)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

result_df = pd.read_excel(get_full_path('patient_analysis_result_val.xlsx'))  # 替换为你的最新文件
plot_df = result_df[result_df['type'].isin(['AML', 'HD'])].copy()

y_true = (plot_df['type'] == 'AML').astype(int)
y_score = plot_df['predicted_ratio']
fpr, tpr, thresholds = roc_curve(y_true, y_score)
J = tpr - fpr
best_thresh = thresholds[np.argmax(J)]

plot_df = plot_df.sort_values('predicted_ratio')
plot_df['color'] = plot_df['type'].map({'AML': 'red', 'HD': 'blue'})

plt.figure(figsize=(18, 8))
ax = plt.gca()

n_patients = len(plot_df)
x = np.arange(n_patients)
width = 0.3

rects1 = ax.bar(x - width, plot_df['actual_ratio'], width, color='lightcoral', edgecolor='red', label='实际比例')
rects2 = ax.bar(x, plot_df['predicted_ratio'], width, color='lightblue', edgecolor='blue', label='预测比例')
rects3 = ax.bar(x + width, plot_df['json_ratio_1'], width, color='lightgreen', edgecolor='green', label='JSON标注比例')

for i, (_, row) in enumerate(plot_df.iterrows()):
    plt.text(x[i], -0.05, row['type'], ha='center', va='top', rotation=90, color=row['color'], fontweight='bold')

def add_value_labels(rects, offset=0.02):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + offset, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

add_value_labels(rects1)
add_value_labels(rects2)
add_value_labels(rects3)

legend_elements = [
    Line2D([0], [0], color='lightcoral', lw=4, label='实际原始细胞比例'),
    Line2D([0], [0], color='lightblue', lw=4, label='预测原始细胞比例'),
    Line2D([0], [0], color='lightgreen', lw=4, label='JSON标注原始细胞比例'),
]

ax.set_ylabel('原始细胞比例', fontsize=12)
ax.set_xlabel('患者类型', fontsize=12, labelpad=50)
ax.set_title('各患者原始细胞比例对比\n(实际 vs 预测 vs JSON标注)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(get_full_path('patient_ratios_comparison_val_with_json.png'), bbox_inches='tight', dpi=300)
plt.close()

print(f"图形已保存，最佳阈值为: {best_thresh:.4f}")
'''