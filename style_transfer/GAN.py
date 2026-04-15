# -*- coding: utf-8 -*-
import os
import sys
import shutil
import subprocess
import random
from glob import glob
from pathlib import Path

############################################
# 1️⃣ 路径配置（按需修改）
############################################

# 是否重新训练
# True  = 重新准备训练集并训练，再推理
# False = 跳过训练，直接使用已有 checkpoint 推理
RUN_TRAIN = False

# 训练集风格域（目标域 B）：支持多个文件夹
TRAIN_DIRS = [
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val",
    # "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/another_train_folder",
]

# 外部测试域（源域 A）
TEST_DIRS = [
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU",
]

# CycleGAN / pytorch-CycleGAN-and-pix2pix 项目根目录
CYCLEGAN_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/style_transfer/pytorch-CycleGAN-and-pix2pix"

# 工作目录
WORK_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer"

# 训练数据目录：用于共享模型训练
TRAIN_DATASET_ROOT = os.path.join(WORK_ROOT, "train_dataset")

# 推理数据目录：每个 test folder 推理时临时使用
INFER_DATASET_ROOT = os.path.join(WORK_ROOT, "infer_dataset")

# 结果目录
RESULTS_DIR = os.path.join(WORK_ROOT, "results")

# 模型名：共享模型
MODEL_NAME = "cell_style_shared"

# checkpoint 根目录（显式指定，避免相对路径混乱）
CHECKPOINTS_DIR = os.path.join(CYCLEGAN_ROOT, "checkpoints")

# 图像后缀
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

# 是否递归扫描子文件夹
RECURSIVE = True

# 随机种子
SEED = 42

# GPU
GPU_ID = "0"   # 设为 "-1" 表示尽量走 CPU

# 训练超参数
BATCH_SIZE = 6
PREPROCESS = "none"
LOAD_SIZE = 256
CROP_SIZE = 256
N_EPOCHS = 20
N_EPOCHS_DECAY = 20
LAMBDA_IDENTITY = 0.5
NO_FLIP = True

# 为避免 A/B 域数量极度不平衡，可设置采样上限；None 表示不限制
MAX_TRAIN_A = 2000
MAX_TRAIN_B = 2000

# 是否平衡两个域的样本数
BALANCE_A_B = True

# 推理时每次保留 fake_B 提取结果
ONLY_SAVE_FAKE_B = True

############################################
# 2️⃣ 工具函数
############################################
def set_seed(seed=42):
    random.seed(seed)

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def resolve_cyclegan_root(root):
    train_py = os.path.join(root, "train.py")
    test_py = os.path.join(root, "test.py")

    if not os.path.isfile(train_py):
        raise FileNotFoundError(
            f"❌ 未找到 train.py: {train_py}\n"
            f"请检查 CYCLEGAN_ROOT 是否指向真正的 pytorch-CycleGAN-and-pix2pix 根目录"
        )
    if not os.path.isfile(test_py):
        raise FileNotFoundError(
            f"❌ 未找到 test.py: {test_py}\n"
            f"请检查 CYCLEGAN_ROOT 是否指向真正的 pytorch-CycleGAN-and-pix2pix 根目录"
        )
    return root

def run_cmd(cmd, cwd=None):
    env = os.environ.copy()

    # 用环境变量控制 GPU，而不是命令行参数
    if GPU_ID is not None and str(GPU_ID).strip() != "" and str(GPU_ID) != "-1":
        env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    print("\n" + "=" * 120)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def normalize_name(path):
    path = os.path.normpath(path)
    name = path.replace(":", "").replace("\\", "_").replace("/", "_").replace(" ", "_")
    return name

def list_images(folder, recursive=True):
    folder = str(folder)
    files = []
    if recursive:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, "**", ext), recursive=True))
    else:
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(folder, ext)))
    return sorted(set(files))

def collect_images_from_folders(folders, recursive=True):
    all_files = []
    valid_folders = []
    for folder in folders:
        if os.path.isdir(folder):
            valid_folders.append(folder)
            imgs = list_images(folder, recursive=recursive)
            all_files.extend(imgs)
        else:
            print(f"⚠️ 跳过，不是有效文件夹: {folder}")
    return valid_folders, sorted(set(all_files))

def sample_files(files, max_n=None):
    if max_n is None:
        return list(files)
    if len(files) <= max_n:
        return list(files)
    return random.sample(files, max_n)

def make_unique_output_name(src_path, root_hint=None):
    """
    生成唯一文件名，避免不同目录下重名文件互相覆盖
    """
    src_path = os.path.normpath(src_path)
    stem = Path(src_path).stem
    suffix = Path(src_path).suffix

    if root_hint is not None:
        try:
            rel = os.path.relpath(src_path, root_hint)
            rel = rel.replace("\\", "_").replace("/", "_")
            rel_stem = os.path.splitext(rel)[0]
            return f"{rel_stem}{suffix}"
        except Exception:
            pass

    parent = Path(src_path).parent.name
    return f"{parent}__{stem}{suffix}"

def copy_images_unique(src_files, dst_dir, root_hint=None):
    safe_mkdir(dst_dir)
    name_map = {}
    used_names = set()

    for f in src_files:
        base_name = make_unique_output_name(f, root_hint=root_hint)

        candidate = base_name
        idx = 1
        while candidate in used_names:
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            candidate = f"{stem}__dup{idx}{suffix}"
            idx += 1

        used_names.add(candidate)
        dst_path = os.path.join(dst_dir, candidate)
        shutil.copy2(f, dst_path)
        name_map[candidate] = f

    return name_map

def ensure_non_empty(files, desc):
    if len(files) == 0:
        raise ValueError(f"❌ 没有找到图像: {desc}")

def print_folder_stats(title, folders):
    print(f"\n📁 {title}")
    for i, folder in enumerate(folders, 1):
        n = len(list_images(folder, recursive=RECURSIVE)) if os.path.isdir(folder) else 0
        print(f"   {i}. {folder}   ({n} images)")

def get_model_ckpt_dir():
    return os.path.join(CHECKPOINTS_DIR, MODEL_NAME)

def get_latest_ga_ckpt():
    return os.path.join(get_model_ckpt_dir(), "latest_net_G_A.pth")

def assert_inference_checkpoint_exists():
    ckpt = get_latest_ga_ckpt()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"❌ 推理所需 checkpoint 不存在: {ckpt}\n"
            f"如果你已经训练完成，请检查 MODEL_NAME/CHECKPOINTS_DIR 是否一致；\n"
            f"如果还没训练，请先把 RUN_TRAIN=True 训练一轮。"
        )

############################################
# 3️⃣ 训练数据准备：共享训练
############################################
def prepare_shared_training_dataset():
    """
    训练一个共享的 A->B 模型
    A域：所有外部测试中心图像（合并）
    B域：所有训练集图像（合并）
    """
    print("\n🚧 Preparing shared training dataset...")

    trainA = os.path.join(TRAIN_DATASET_ROOT, "trainA")
    trainB = os.path.join(TRAIN_DATASET_ROOT, "trainB")
    testA = os.path.join(TRAIN_DATASET_ROOT, "testA")
    testB = os.path.join(TRAIN_DATASET_ROOT, "testB")

    clear_and_make(trainA)
    clear_and_make(trainB)
    clear_and_make(testA)
    clear_and_make(testB)

    valid_train_dirs, train_imgs_all = collect_images_from_folders(TRAIN_DIRS, recursive=RECURSIVE)
    valid_test_dirs, test_imgs_all = collect_images_from_folders(TEST_DIRS, recursive=RECURSIVE)

    ensure_non_empty(train_imgs_all, "TRAIN_DIRS")
    ensure_non_empty(test_imgs_all, "TEST_DIRS")

    print_folder_stats("有效训练集文件夹（目标域 B）", valid_train_dirs)
    print_folder_stats("有效外部测试文件夹（源域 A）", valid_test_dirs)

    # 先各自采样上限
    train_imgs = sample_files(train_imgs_all, MAX_TRAIN_B)
    test_imgs = sample_files(test_imgs_all, MAX_TRAIN_A)

    # 再按需平衡 A/B
    if BALANCE_A_B:
        n = min(len(train_imgs), len(test_imgs))
        train_imgs = sample_files(train_imgs, n)
        test_imgs = sample_files(test_imgs, n)

    copy_images_unique(test_imgs, trainA)
    copy_images_unique(train_imgs, trainB)

    print("\n✅ Shared training dataset prepared")
    print(f"   trainA (all external domains): {len(test_imgs)}")
    print(f"   trainB (all train domains)   : {len(train_imgs)}")
    print(f"   dataroot                     : {TRAIN_DATASET_ROOT}")

############################################
# 4️⃣ 推理数据准备：按单个 test folder 输出
############################################
def prepare_infer_dataset_for_one_test(test_dir):
    """
    推理阶段不再重新训练，只替换 testA
    """
    print(f"\n📂 Preparing inference dataset for: {test_dir}")

    testA = os.path.join(INFER_DATASET_ROOT, "testA")
    testB = os.path.join(INFER_DATASET_ROOT, "testB")

    clear_and_make(testA)
    clear_and_make(testB)

    test_imgs = list_images(test_dir, recursive=RECURSIVE)
    ensure_non_empty(test_imgs, test_dir)

    name_map = copy_images_unique(test_imgs, testA, root_hint=test_dir)

    print(f"✅ testA: {len(test_imgs)}")
    return name_map

############################################
# 5️⃣ 训练共享 CycleGAN
############################################
def train_shared_cyclegan():
    cyclegan_root = resolve_cyclegan_root(CYCLEGAN_ROOT)
    train_script = os.path.join(cyclegan_root, "train.py")

    safe_mkdir(CHECKPOINTS_DIR)

    cmd = [
        sys.executable, train_script,
        "--dataroot", TRAIN_DATASET_ROOT,
        "--name", MODEL_NAME,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--model", "cycle_gan",
        "--direction", "AtoB",
        "--batch_size", str(BATCH_SIZE),
        "--preprocess", PREPROCESS,
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--n_epochs", str(N_EPOCHS),
        "--n_epochs_decay", str(N_EPOCHS_DECAY),
        "--lambda_identity", str(LAMBDA_IDENTITY),
    ]

    if NO_FLIP:
        cmd.append("--no_flip")

    run_cmd(cmd, cwd=cyclegan_root)

############################################
# 6️⃣ 使用共享模型做推理
############################################
def test_shared_cyclegan(phase_name):
    """
    用 phase_name 区分不同 test folder 的输出
    """
    cyclegan_root = resolve_cyclegan_root(CYCLEGAN_ROOT)
    test_script = os.path.join(cyclegan_root, "test.py")

    assert_inference_checkpoint_exists()

    cmd = [
        sys.executable, test_script,
        "--dataroot", INFER_DATASET_ROOT,
        "--name", MODEL_NAME,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--model", "test",
        "--model_suffix", "_A",          # 加载 latest_net_G_A.pth
        "--dataset_mode", "single",
        "--direction", "AtoB",
        "--num_test", "999999",
        "--preprocess", PREPROCESS,
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--results_dir", RESULTS_DIR,
        "--phase", phase_name,
        "--netG", "resnet_9blocks",
        "--no_dropout",                  # 关键补丁：和训练保持一致
        "--batch_size", "64",
        "--num_threads", "16",
    ]

    if NO_FLIP:
        cmd.append("--no_flip")

    run_cmd(cmd, cwd=cyclegan_root)

############################################
# 7️⃣ 整理输出结果
############################################
def extract_fake_b_images(test_dir, phase_name, name_map):
    """
    从 results 中提取 fake_B，并尽量恢复可读文件名
    """
    src_dir = os.path.join(RESULTS_DIR, MODEL_NAME, f"{phase_name}_latest", "images")
    test_name = normalize_name(test_dir)
    dst_dir = os.path.join(RESULTS_DIR, MODEL_NAME, "translated", test_name)

    if not os.path.exists(src_dir):
        print(f"⚠️ 未找到结果目录: {src_dir}")
        return

    clear_and_make(dst_dir)

    fake_files = sorted(
        glob(os.path.join(src_dir, "*_fake.png")) +
        glob(os.path.join(src_dir, "*_fake_B.png"))
    )

    if len(fake_files) == 0:
        print("⚠️ 未发现 *_fake.png 或 *_fake_B.png，兜底复制全部结果")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"✅ 已保存到: {dst_dir}")
        return

    for fake_path in fake_files:
        fake_name = os.path.basename(fake_path)

        if fake_name.endswith("_fake_B.png"):
            base = fake_name[:-len("_fake_B.png")]
        elif fake_name.endswith("_fake.png"):
            base = fake_name[:-len("_fake.png")]
        else:
            base = Path(fake_name).stem

        out_name = f"{base}.png"
        dst_path = os.path.join(dst_dir, out_name)
        shutil.copy2(fake_path, dst_path)

    print(f"✅ fake_B 结果已保存到: {dst_dir}")

############################################
# 8️⃣ 主流程
############################################
def main():
    set_seed(SEED)

    safe_mkdir(WORK_ROOT)
    safe_mkdir(RESULTS_DIR)
    safe_mkdir(CHECKPOINTS_DIR)

    print("=" * 120)
    print("Style Transfer Pipeline v2")
    print("=" * 120)
    print(f"RUN_TRAIN        : {RUN_TRAIN}")
    print(f"MODEL_NAME       : {MODEL_NAME}")
    print(f"CYCLEGAN_ROOT    : {CYCLEGAN_ROOT}")
    print(f"CHECKPOINTS_DIR  : {CHECKPOINTS_DIR}")
    print(f"TRAIN_DATASET    : {TRAIN_DATASET_ROOT}")
    print(f"INFER_DATASET    : {INFER_DATASET_ROOT}")
    print(f"RESULTS_DIR      : {RESULTS_DIR}")
    print(f"GPU_ID           : {GPU_ID}")
    print(f"RECURSIVE        : {RECURSIVE}")
    print(f"BALANCE_A_B      : {BALANCE_A_B}")
    print(f"MAX_TRAIN_A      : {MAX_TRAIN_A}")
    print(f"MAX_TRAIN_B      : {MAX_TRAIN_B}")
    print("=" * 120)

    # 1) 训练（可跳过）
    if RUN_TRAIN:
        prepare_shared_training_dataset()
        train_shared_cyclegan()
    else:
        print("⏭️ 跳过训练，直接使用已有 checkpoint 做推理")
        print(f"🔍 期望 checkpoint: {get_latest_ga_ckpt()}")
        assert_inference_checkpoint_exists()

    # 2) 对每个外部测试文件夹分别推理
    valid_test_dirs, _ = collect_images_from_folders(TEST_DIRS, recursive=RECURSIVE)
    if len(valid_test_dirs) == 0:
        raise ValueError("❌ 没有可用的 TEST_DIRS")

    for i, test_dir in enumerate(valid_test_dirs, 1):
        test_name = normalize_name(test_dir)
        phase_name = f"test_{i}_{test_name}"

        print("\n" + "#" * 120)
        print(f"🚀 Inference for test folder [{i}/{len(valid_test_dirs)}]: {test_dir}")
        print("#" * 120)

        name_map = prepare_infer_dataset_for_one_test(test_dir)
        test_shared_cyclegan(phase_name=phase_name)
        extract_fake_b_images(test_dir, phase_name, name_map)

    print("\n🎉 All test folders processed successfully.")

############################################
# 9️⃣ 入口
############################################
if __name__ == "__main__":
    main()

# # -*- coding: utf-8 -*-
# import os
# import sys
# import shutil
# import subprocess
# import random
# from glob import glob
# from pathlib import Path

# ############################################
# # 1️⃣ 路径配置（按需修改）
# ############################################

# # 训练集风格域（目标域 B）：现在支持多个文件夹
# TRAIN_DIRS = [
#     "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/train",
#     "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/val",
#     # "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/another_train_folder",
# ]

# # 外部测试域（源域 A）：可以多个中心一起用于训练共享风格迁移模型
# TEST_DIRS = [
#     "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_BJH",
#     "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_FXH_noALL",
#     "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell_260323/test_TJMU",
# ]

# # CycleGAN / pytorch-CycleGAN-and-pix2pix 项目根目录
# # 即 train.py / test.py 所在目录
# CYCLEGAN_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/style_transfer/pytorch-CycleGAN-and-pix2pix"

# # 工作目录
# WORK_ROOT = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer"

# # 训练数据目录：用于共享模型训练
# TRAIN_DATASET_ROOT = os.path.join(WORK_ROOT, "train_dataset")

# # 推理数据目录：每个 test folder 推理时临时使用
# INFER_DATASET_ROOT = os.path.join(WORK_ROOT, "infer_dataset")

# # 结果目录
# RESULTS_DIR = os.path.join(WORK_ROOT, "results")

# # 模型名：现在只训练一个共享模型
# MODEL_NAME = "cell_style_shared"

# # 图像后缀
# IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

# # 是否递归扫描子文件夹
# RECURSIVE = True

# # 随机种子
# SEED = 42

# # GPU
# GPU_ID = "0"   # 设为 "-1" 表示尽量走 CPU

# # 训练超参数
# BATCH_SIZE = 1
# PREPROCESS = "none"         # 对单细胞更稳，尽量少做随机裁剪破坏形态
# LOAD_SIZE = 256
# CROP_SIZE = 256
# LAMBDA_IDENTITY = 0.5       # 保持内容结构，减少“风格过强导致形变”
# NO_FLIP = True              # 单细胞风格迁移通常不建议乱翻转

# # 为避免 A/B 域数量极度不平衡，可设置采样上限；None 表示不限制
# BATCH_SIZE = 6
# MAX_TRAIN_A = 2000
# MAX_TRAIN_B = 2000
# N_EPOCHS = 20
# N_EPOCHS_DECAY = 20

# # 是否平衡两个域的样本数（推荐 True）
# BALANCE_A_B = True

# # 推理时每次保留 fake_B 提取结果
# ONLY_SAVE_FAKE_B = True

# ############################################
# # 2️⃣ 工具函数
# ############################################
# def set_seed(seed=42):
#     random.seed(seed)

# def safe_mkdir(path):
#     os.makedirs(path, exist_ok=True)

# def clear_and_make(path):
#     if os.path.exists(path):
#         shutil.rmtree(path)
#     os.makedirs(path, exist_ok=True)

# def resolve_cyclegan_root(root):
#     train_py = os.path.join(root, "train.py")
#     test_py = os.path.join(root, "test.py")

#     if not os.path.isfile(train_py):
#         raise FileNotFoundError(
#             f"❌ 未找到 train.py: {train_py}\n"
#             f"请检查 CYCLEGAN_ROOT 是否指向真正的 pytorch-CycleGAN-and-pix2pix 根目录"
#         )
#     if not os.path.isfile(test_py):
#         raise FileNotFoundError(
#             f"❌ 未找到 test.py: {test_py}\n"
#             f"请检查 CYCLEGAN_ROOT 是否指向真正的 pytorch-CycleGAN-and-pix2pix 根目录"
#         )
#     return root

# def run_cmd(cmd, cwd=None):
#     env = os.environ.copy()

#     # 用环境变量控制 GPU，而不是命令行参数
#     if GPU_ID is not None and str(GPU_ID).strip() != "" and str(GPU_ID) != "-1":
#         env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
#     else:
#         # 显式禁用 GPU
#         env["CUDA_VISIBLE_DEVICES"] = ""

#     print("\n" + "=" * 120)
#     print("Running command:")
#     print(" ".join(cmd))
#     print("=" * 120)
#     subprocess.run(cmd, cwd=cwd, env=env, check=True)

# def normalize_name(path):
#     path = os.path.normpath(path)
#     name = path.replace(":", "").replace("\\", "_").replace("/", "_").replace(" ", "_")
#     return name

# def list_images(folder, recursive=True):
#     folder = str(folder)
#     files = []
#     if recursive:
#         for ext in IMG_EXTS:
#             files.extend(glob(os.path.join(folder, "**", ext), recursive=True))
#     else:
#         for ext in IMG_EXTS:
#             files.extend(glob(os.path.join(folder, ext)))
#     return sorted(set(files))

# def collect_images_from_folders(folders, recursive=True):
#     all_files = []
#     valid_folders = []
#     for folder in folders:
#         if os.path.isdir(folder):
#             valid_folders.append(folder)
#             imgs = list_images(folder, recursive=recursive)
#             all_files.extend(imgs)
#         else:
#             print(f"⚠️ 跳过，不是有效文件夹: {folder}")
#     return valid_folders, sorted(set(all_files))

# def sample_files(files, max_n=None):
#     if max_n is None:
#         return list(files)
#     if len(files) <= max_n:
#         return list(files)
#     return random.sample(files, max_n)

# def make_unique_output_name(src_path, root_hint=None):
#     """
#     生成唯一文件名，避免不同目录下重名文件互相覆盖
#     """
#     src_path = os.path.normpath(src_path)
#     stem = Path(src_path).stem
#     suffix = Path(src_path).suffix

#     if root_hint is not None:
#         try:
#             rel = os.path.relpath(src_path, root_hint)
#             rel = rel.replace("\\", "_").replace("/", "_")
#             rel_stem = os.path.splitext(rel)[0]
#             return f"{rel_stem}{suffix}"
#         except Exception:
#             pass

#     parent = Path(src_path).parent.name
#     return f"{parent}__{stem}{suffix}"

# def copy_images_unique(src_files, dst_dir, root_hint=None):
#     safe_mkdir(dst_dir)
#     name_map = {}
#     used_names = set()

#     for f in src_files:
#         base_name = make_unique_output_name(f, root_hint=root_hint)

#         # 防止仍然重名
#         candidate = base_name
#         idx = 1
#         while candidate in used_names:
#             stem = Path(base_name).stem
#             suffix = Path(base_name).suffix
#             candidate = f"{stem}__dup{idx}{suffix}"
#             idx += 1

#         used_names.add(candidate)
#         dst_path = os.path.join(dst_dir, candidate)
#         shutil.copy2(f, dst_path)
#         name_map[candidate] = f

#     return name_map

# def ensure_non_empty(files, desc):
#     if len(files) == 0:
#         raise ValueError(f"❌ 没有找到图像: {desc}")

# def print_folder_stats(title, folders):
#     print(f"\n📁 {title}")
#     for i, folder in enumerate(folders, 1):
#         n = len(list_images(folder, recursive=RECURSIVE)) if os.path.isdir(folder) else 0
#         print(f"   {i}. {folder}   ({n} images)")

# ############################################
# # 3️⃣ 训练数据准备：共享训练
# ############################################
# def prepare_shared_training_dataset():
#     """
#     训练一个共享的 A->B 模型
#     A域：所有外部测试中心图像（合并）
#     B域：所有训练集图像（合并）
#     """
#     print("\n🚧 Preparing shared training dataset...")

#     trainA = os.path.join(TRAIN_DATASET_ROOT, "trainA")
#     trainB = os.path.join(TRAIN_DATASET_ROOT, "trainB")
#     testA = os.path.join(TRAIN_DATASET_ROOT, "testA")
#     testB = os.path.join(TRAIN_DATASET_ROOT, "testB")

#     clear_and_make(trainA)
#     clear_and_make(trainB)
#     clear_and_make(testA)
#     clear_and_make(testB)

#     valid_train_dirs, train_imgs_all = collect_images_from_folders(TRAIN_DIRS, recursive=RECURSIVE)
#     valid_test_dirs, test_imgs_all = collect_images_from_folders(TEST_DIRS, recursive=RECURSIVE)

#     ensure_non_empty(train_imgs_all, "TRAIN_DIRS")
#     ensure_non_empty(test_imgs_all, "TEST_DIRS")

#     print_folder_stats("有效训练集文件夹（目标域 B）", valid_train_dirs)
#     print_folder_stats("有效外部测试文件夹（源域 A）", valid_test_dirs)

#     # 先各自采样上限
#     train_imgs = sample_files(train_imgs_all, MAX_TRAIN_B)
#     test_imgs = sample_files(test_imgs_all, MAX_TRAIN_A)

#     # 再按需平衡 A/B
#     if BALANCE_A_B:
#         n = min(len(train_imgs), len(test_imgs))
#         train_imgs = sample_files(train_imgs, n)
#         test_imgs = sample_files(test_imgs, n)

#     copy_images_unique(test_imgs, trainA)
#     copy_images_unique(train_imgs, trainB)

#     print("\n✅ Shared training dataset prepared")
#     print(f"   trainA (all external domains): {len(test_imgs)}")
#     print(f"   trainB (all train domains)   : {len(train_imgs)}")
#     print(f"   dataroot                     : {TRAIN_DATASET_ROOT}")

# ############################################
# # 4️⃣ 推理数据准备：按单个 test folder 输出
# ############################################
# def prepare_infer_dataset_for_one_test(test_dir):
#     """
#     推理阶段不再重新训练，只替换 testA
#     """
#     print(f"\n📂 Preparing inference dataset for: {test_dir}")

#     testA = os.path.join(INFER_DATASET_ROOT, "testA")
#     testB = os.path.join(INFER_DATASET_ROOT, "testB")

#     clear_and_make(testA)
#     clear_and_make(testB)

#     test_imgs = list_images(test_dir, recursive=RECURSIVE)
#     ensure_non_empty(test_imgs, test_dir)

#     name_map = copy_images_unique(test_imgs, testA, root_hint=test_dir)

#     print(f"✅ testA: {len(test_imgs)}")
#     return name_map

# ############################################
# # 5️⃣ 训练共享 CycleGAN
# ############################################
# def train_shared_cyclegan():
#     cyclegan_root = resolve_cyclegan_root(CYCLEGAN_ROOT)
#     train_script = os.path.join(cyclegan_root, "train.py")

#     cmd = [
#         sys.executable, train_script,
#         "--dataroot", TRAIN_DATASET_ROOT,
#         "--name", MODEL_NAME,
#         "--model", "cycle_gan",
#         "--direction", "AtoB",
#         "--batch_size", str(BATCH_SIZE),
#         "--preprocess", PREPROCESS,
#         "--load_size", str(LOAD_SIZE),
#         "--crop_size", str(CROP_SIZE),
#         "--n_epochs", str(N_EPOCHS),
#         "--n_epochs_decay", str(N_EPOCHS_DECAY),
#         "--lambda_identity", str(LAMBDA_IDENTITY),
#     ]

#     if NO_FLIP:
#         cmd.append("--no_flip")

#     run_cmd(cmd, cwd=cyclegan_root)

# ############################################
# # 6️⃣ 使用共享模型做推理
# ############################################
# def test_shared_cyclegan(phase_name):
#     """
#     用 phase_name 区分不同 test folder 的输出
#     """
#     cyclegan_root = resolve_cyclegan_root(CYCLEGAN_ROOT)
#     test_script = os.path.join(cyclegan_root, "test.py")

#     cmd = [
#         sys.executable, test_script,
#         "--dataroot", INFER_DATASET_ROOT,
#         "--name", MODEL_NAME,
#         "--model", "test",
#         "--netG", "resnet_9blocks",
#         "--dataset_mode", "single",
#         "--direction", "AtoB",
#         "--num_test", "999999",
#         "--preprocess", PREPROCESS,
#         "--load_size", str(LOAD_SIZE),
#         "--crop_size", str(CROP_SIZE),
#         "--results_dir", RESULTS_DIR,
#         "--phase", phase_name,
#     ]

#     if NO_FLIP:
#         cmd.append("--no_flip")

#     run_cmd(cmd, cwd=cyclegan_root)

# ############################################
# # 7️⃣ 整理输出结果
# ############################################
# def extract_fake_b_images(test_dir, phase_name, name_map):
#     """
#     从 results 中提取 fake_B，并尽量恢复可读文件名
#     """
#     src_dir = os.path.join(RESULTS_DIR, MODEL_NAME, f"{phase_name}_latest", "images")
#     test_name = normalize_name(test_dir)
#     dst_dir = os.path.join(RESULTS_DIR, MODEL_NAME, "translated", test_name)

#     if not os.path.exists(src_dir):
#         print(f"⚠️ 未找到结果目录: {src_dir}")
#         return

#     clear_and_make(dst_dir)

#     fake_files = sorted(glob(os.path.join(src_dir, "*_fake.png")) +
#                         glob(os.path.join(src_dir, "*_fake_B.png")))

#     if len(fake_files) == 0:
#         # 如果 test.py 版本输出命名不同，兜底复制全部
#         print("⚠️ 未发现 *_fake.png 或 *_fake_B.png，兜底复制全部结果")
#         shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
#         print(f"✅ 已保存到: {dst_dir}")
#         return

#     for fake_path in fake_files:
#         fake_name = os.path.basename(fake_path)

#         if fake_name.endswith("_fake_B.png"):
#             base = fake_name[:-len("_fake_B.png")]
#         elif fake_name.endswith("_fake.png"):
#             base = fake_name[:-len("_fake.png")]
#         else:
#             base = Path(fake_name).stem

#         out_name = f"{base}.png"
#         dst_path = os.path.join(dst_dir, out_name)
#         shutil.copy2(fake_path, dst_path)

#     print(f"✅ fake_B 结果已保存到: {dst_dir}")

# ############################################
# # 8️⃣ 主流程
# ############################################
# def main():
#     set_seed(SEED)

#     safe_mkdir(WORK_ROOT)
#     safe_mkdir(RESULTS_DIR)

#     print("=" * 120)
#     print("Style Transfer Pipeline v2")
#     print("=" * 120)
#     print(f"MODEL_NAME       : {MODEL_NAME}")
#     print(f"CYCLEGAN_ROOT    : {CYCLEGAN_ROOT}")
#     print(f"TRAIN_DATASET    : {TRAIN_DATASET_ROOT}")
#     print(f"INFER_DATASET    : {INFER_DATASET_ROOT}")
#     print(f"RESULTS_DIR      : {RESULTS_DIR}")
#     print(f"GPU_ID           : {GPU_ID}")
#     print(f"RECURSIVE        : {RECURSIVE}")
#     print(f"BALANCE_A_B      : {BALANCE_A_B}")
#     print(f"MAX_TRAIN_A      : {MAX_TRAIN_A}")
#     print(f"MAX_TRAIN_B      : {MAX_TRAIN_B}")
#     print("=" * 120)

#     # 1) 准备共享训练集并训练一次
#     prepare_shared_training_dataset()
#     train_shared_cyclegan()

#     # 2) 对每个外部测试文件夹分别推理
#     valid_test_dirs, _ = collect_images_from_folders(TEST_DIRS, recursive=RECURSIVE)
#     if len(valid_test_dirs) == 0:
#         raise ValueError("❌ 没有可用的 TEST_DIRS")

#     for i, test_dir in enumerate(valid_test_dirs, 1):
#         test_name = normalize_name(test_dir)
#         phase_name = f"test_{i}_{test_name}"

#         print("\n" + "#" * 120)
#         print(f"🚀 Inference for test folder [{i}/{len(valid_test_dirs)}]: {test_dir}")
#         print("#" * 120)

#         name_map = prepare_infer_dataset_for_one_test(test_dir)
#         test_shared_cyclegan(phase_name=phase_name)
#         extract_fake_b_images(test_dir, phase_name, name_map)

#     print("\n🎉 All test folders processed successfully.")

# ############################################
# # 9️⃣ 入口
# ############################################
# if __name__ == "__main__":
#     main()