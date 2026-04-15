# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ============================== Big label 映射 ==============================
# 规则：
# 1类：N, N1, M, M1, R, R1, J, J1
# 其余已知标签 -> 2
# 未知标签 -> 0
def map_big_label(raw_label: str) -> int:
    raw_label = str(raw_label).strip().upper()

    positive_group = {"N", "N1", "M", "M1", "R", "R1", "J", "J1"}
    negative_group = {
        "V", "0",
        "N0", "N2", "N3", "N4", "N5",
        "E", "B", "E1", "B1",
        "M0", "M2",
        "R2", "R3",
        "J2", "J3", "J4",
        "P", "P1", "P2", "P3",
        "L", "L1", "L2", "L3", "L4"
    }

    if raw_label in positive_group:
        return 1
    elif raw_label in negative_group:
        return 2
    else:
        return 0


# ============================== Small label 映射（19类） ==============================
# 19类如下：
# 0  -> N
# 1  -> N1
# 2  -> N2
# 3  -> N3
# 4  -> N4
# 5  -> N5
# 6  -> E-E/E1
# 7  -> B-B/B1
# 8  -> M-M0/M
# 9  -> M1
# 10 -> M2
# 11 -> R
# 12 -> R1
# 13 -> R2
# 14 -> R3
# 15 -> J-J/J1/J2/J3/J4
# 16 -> L-L/L1/L2/L3/L4
# 17 -> P-P/P1/P2/P3
# 18 -> 0-OTHERS
# def map_small_label(raw_label: str) -> int:
#     raw_label = str(raw_label).strip().upper()

#     if raw_label == "N":
#         return 0
#     elif raw_label == "N1":
#         return 1
#     elif raw_label == "N2":
#         return 2
#     elif raw_label == "N3":
#         return 3
#     elif raw_label == "N4":
#         return 4
#     elif raw_label == "N5":
#         return 5

#     elif raw_label in {"E", "E1"}:
#         return 6
#     elif raw_label in {"B", "B1"}:
#         return 7

#     elif raw_label in {"M", "M0"}:
#         return 8
#     elif raw_label == "M1":
#         return 9
#     elif raw_label == "M2":
#         return 10

#     elif raw_label == "R":
#         return 11
#     elif raw_label == "R1":
#         return 12
#     elif raw_label == "R2":
#         return 13
#     elif raw_label == "R3":
#         return 14

#     elif raw_label in {"J", "J1", "J2", "J3", "J4"}:
#         return 15

#     elif raw_label in {"L", "L1", "L2", "L3", "L4"}:
#         return 16

#     elif raw_label in {"P", "P1", "P2", "P3"}:
#         return 17

#     else:
#         return 18

def map_small_label(raw_label: str) -> int:
    raw_label = str(raw_label).strip().upper()

    if raw_label == "N":
        return 1
    elif raw_label == "N1":
        return 2
    elif raw_label == "N2":
        return 3
    elif raw_label == "N3":
        return 4
    elif raw_label == "N4":
        return 5
    elif raw_label == "N5":
        return 6

    elif raw_label in {"E", "E1"}:
        return 7

    elif raw_label in {"M", "M0"}:
        return 8
    elif raw_label == "M1":
        return 9
    elif raw_label == "M2":
        return 10

    elif raw_label == "R":
        return 11
    elif raw_label == "R1":
        return 12
    elif raw_label == "R2":
        return 13
    elif raw_label == "R3":
        return 14
    
    elif raw_label in {"L", "L1", "L2", "L3", "L4"}:
        return 15

    else:
        return 0

# ============================== 生成标签文件 ==============================
def DataTxtGenerator(output_dir):
    output_dir = Path(output_dir)

    split_names = [
        "beta_0p01",
    ]

    for split_name in split_names:
        img_dir = output_dir / split_name

        if not img_dir.exists():
            print(f"ℹ️ {split_name} 目录不存在，跳过标签生成。")
            continue

        imgs = list(img_dir.rglob("*.png"))
        imgs = [p for p in imgs if ".ipynb_checkpoints" not in str(p)]

        txt_path = output_dir / f"{split_name}_labels_16.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for img_path in tqdm(imgs, desc=f"生成标签: {split_name}", unit="line"):
                img_name = img_path.name

                # 取文件名最后一个下划线后的标签
                # 例如：PKUPH-106-10_000_P2.png -> P2
                raw_label = img_name.split(".")[0].split("_")[-1].strip().upper()

                label_big = map_big_label(raw_label)
                label_small = map_small_label(raw_label)

                f.write(f"{str(img_path)} {label_big} {label_small}\n")

        print(f"✅ 已生成: {txt_path}")


if __name__ == "__main__":
    output_dir = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer_fda/test_TJMU"
    DataTxtGenerator(output_dir)