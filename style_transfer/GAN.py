import os
import shutil
import subprocess
from glob import glob

############################################
# 1️⃣ 路径配置（改这里）
############################################
TRAIN_DIR = "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell/train"   # 目标风格：训练集单细胞图像

# 多个分散的 test 文件夹，直接逐个写路径
TEST_DIRS = [
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell/test_BEPH",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell/test_BJH",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell/test_FXH_noALL",
    "/root/autodl-tmp/projects/myq/SingleCellProject/dataset/singlecell/test_TJMU"
    # r"D:/your/other/test_folder",
]

DATASET_ROOT = r"/root/autodl-tmp/projects/myq/SingleCellProject/dataset/style_transfer"
RESULTS_DIR = r"./results"
NAME_PREFIX = "cell_style"

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

############################################
# 2️⃣ 工具函数
############################################
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files)

def copy_images(src_files, dst_dir):
    safe_mkdir(dst_dir)
    for f in src_files:
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

def run_cmd(cmd, cwd=None):
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=cwd, check=True)

def normalize_name(path):
    """
    将路径转成适合当模型名/结果目录名的字符串
    """
    path = os.path.normpath(path)
    name = path.replace(":", "").replace("\\", "_").replace("/", "_").replace(" ", "_")
    return name

############################################
# 3️⃣ 检查 test 文件夹列表
############################################
def get_test_folders():
    valid_folders = []
    for folder in TEST_DIRS:
        if os.path.isdir(folder):
            valid_folders.append(folder)
        else:
            print(f"⚠️ 跳过，不是有效文件夹: {folder}")
    return valid_folders

############################################
# 4️⃣ 为单个 test 文件夹准备数据
############################################
def prepare_dataset_for_one_test(test_dir):
    """
    datasets/cell/
        trainA <- 当前 test 图像
        trainB <- train 图像（目标风格）
        testA  <- 当前 test 图像（推理输入）
    """
    print(f"\n📂 Preparing dataset for: {test_dir}")

    trainA = os.path.join(DATASET_ROOT, "trainA")
    trainB = os.path.join(DATASET_ROOT, "trainB")
    testA = os.path.join(DATASET_ROOT, "testA")
    testB = os.path.join(DATASET_ROOT, "testB")

    clear_and_make(trainA)
    clear_and_make(trainB)
    clear_and_make(testA)
    clear_and_make(testB)

    train_imgs = list_images(TRAIN_DIR)
    test_imgs = list_images(test_dir)

    if len(train_imgs) == 0:
        raise ValueError(f"❌ TRAIN_DIR 中没有找到图像: {TRAIN_DIR}")
    if len(test_imgs) == 0:
        raise ValueError(f"❌ test 文件夹中没有找到图像: {test_dir}")

    copy_images(test_imgs, trainA)
    copy_images(train_imgs, trainB)
    copy_images(test_imgs, testA)

    print(f"✅ trainA: {len(test_imgs)} images")
    print(f"✅ trainB: {len(train_imgs)} images")
    print(f"✅ testA : {len(test_imgs)} images")

############################################
# 5️⃣ 训练 CycleGAN
############################################
def train_cyclegan(model_name, dataroot=DATASET_ROOT, gpu_id="0"):
    cmd = [
        "python", "train.py",
        "--dataroot", dataroot,
        "--name", model_name,
        "--model", "cycle_gan",
        "--direction", "AtoB",
        "--gpu_ids", gpu_id,
        "--batch_size", "1",
        "--preprocess", "resize_and_crop",
        "--load_size", "286",
        "--crop_size", "256",
        "--display_id", "-1"
    ]
    run_cmd(cmd)

############################################
# 6️⃣ 推理
############################################
def test_cyclegan(model_name, dataroot=DATASET_ROOT, gpu_id="0", num_test=999999):
    cmd = [
        "python", "test.py",
        "--dataroot", dataroot,
        "--name", model_name,
        "--model", "cycle_gan",
        "--direction", "AtoB",
        "--gpu_ids", gpu_id,
        "--num_test", str(num_test),
        "--preprocess", "resize_and_crop",
        "--load_size", "256",
        "--crop_size", "256",
        "--results_dir", RESULTS_DIR
    ]
    run_cmd(cmd)

############################################
# 7️⃣ 备份结果
############################################
def backup_results(test_dir, model_name):
    src_dir = os.path.join(RESULTS_DIR, model_name, "test_latest", "images")
    test_name = normalize_name(test_dir)
    dst_dir = os.path.join(RESULTS_DIR, f"{model_name}_{test_name}")

    if not os.path.exists(src_dir):
        print(f"⚠️ 未找到结果目录: {src_dir}")
        return

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"✅ 结果已保存到: {dst_dir}")

############################################
# 8️⃣ 主流程
############################################
def main():
    test_folders = get_test_folders()

    if len(test_folders) == 0:
        raise ValueError("❌ 没有可用的 test 文件夹，请检查 TEST_DIRS 配置")

    print(f"🔍 Found {len(test_folders)} valid test folders:")
    for i, folder in enumerate(test_folders, 1):
        print(f"   {i}. {folder}")

    for test_dir in test_folders:
        test_name = normalize_name(test_dir)
        model_name = f"{NAME_PREFIX}_{test_name}"

        print("\n" + "#" * 100)
        print(f"🚀 Processing test folder: {test_dir}")
        print("#" * 100)

        prepare_dataset_for_one_test(test_dir)
        train_cyclegan(model_name=model_name, dataroot=DATASET_ROOT, gpu_id="0")
        test_cyclegan(model_name=model_name, dataroot=DATASET_ROOT, gpu_id="0")
        backup_results(test_dir, model_name)

    print("\n🎉 All test folders have been processed successfully.")

############################################
# 9️⃣ 入口
############################################
if __name__ == "__main__":
    main()