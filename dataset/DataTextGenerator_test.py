# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm



# ============================== 类别映射 ==============================
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

cell_dict_small = {
    "V": 0, "0": 0, "N": 1, "N1": 1, "N0": 2, "N2": 2, "N3": 2, "N4": 2, "N5": 2,
    "E": 2, "B": 2, "E1": 2, "B1": 2, "M": 3, "M1": 3, "M0": 4, "M2": 4,
    "R": 5, "R1": 5, "R2": 6, "R3": 6, "J": 7, "J1": 7, "J2": 8, "J3": 8, "J4": 8,
    "L": 9, "L1": 9, "L2": 9, "L3": 9, "L4": 9, "P": 10, "P1": 10, "P2": 10, "P3": 10
}


# ============================== 生成标签文件 ==============================
def DataTxtGenerator(output_dir):
    for split_name in ['test_FXH_noALL', 'test_BJH', 'test_TJMU','train', 'val']:
        img_dir = os.path.join(output_dir, split_name)
        if not os.path.exists(img_dir):
            print(f"ℹ️ {split_name} 目录不存在，跳过标签生成。")
            continue
            
        # 同样过滤 .ipynb_checkpoints
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.png') and '.ipynb_checkpoints' not in f]
        txt_path = os.path.join(output_dir, f"{split_name}_labels.txt")

        with open(txt_path, 'w', encoding='utf-8') as f:
            for img_name in tqdm(imgs, desc=f"生成标签: {split_name}", unit="line"):
                # base_name = img_name.rsplit('_', 1)[0]
                # parts = base_name.split('_')
                # raw_label = parts[-1]
                raw_label = img_name.split('.')[0].split('_')[-1]  
          

                label_big = cell_dict_big.get(raw_label, 0)
                label_small = cell_dict_small.get(raw_label, 0)

                img_path = os.path.join(output_dir, split_name, img_name)
                f.write(f"{img_path} {label_big} {label_small}\n")


if __name__ == "__main__":
    output_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323"
    DataTxtGenerator(output_dir)