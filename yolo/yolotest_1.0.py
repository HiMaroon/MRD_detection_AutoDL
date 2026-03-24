
# from ultralytics import YOLO
# from pathlib import Path
# import json
# import os
# from tqdm import tqdm

# # ===================== 配置参数 =====================

# MODEL_PATH = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/cellseg/260323_MAIN_yolo11m/weights/best.pt"

# DATA_ROOTS = {
#     "test_FXH_noALL": Path("/root/autodl-tmp/data/FXH_imgs_noALL_260318"),
#     "test_BJH": Path("/root/autodl-tmp/data/BJH_imgs_260211"),
#     "test_TJMU": Path("/root/autodl-tmp/data/TJMU_imgs_260318"),
#     "train": Path("/root/autodl-tmp/data/MAIN_imgs_split_260323/Train"),
#     "val": Path("/root/autodl-tmp/data/MAIN_imgs_split_260323/Val"),
# }

# PRED_ROOT = Path("/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/")

# target_sets = ["test_FXH_noALL",
#                "test_BJH",
#                "test_TJMU",
#                "train",
#                "val"]

# # ===================== 执行推理 =====================

# def run_test():
#     # 1. 加载模型 (只在循环外加载一次)
#     print(f"🚀 正在加载模型：{MODEL_PATH}")
#     model = YOLO(MODEL_PATH)
#     # 预热一下 (可选，防止第一张图慢)
#     # model.predict(source=0, imgsz=320, verbose=False) 

#     for target_set in target_sets:
#         input_dir = DATA_ROOTS[target_set]
#         output_base = PRED_ROOT / target_set

#         print(f"🔍 正在扫描 {target_set} 目录下的图片...")
#         # 提前收集所有图片路径
#         image_list = list(input_dir.rglob("*.jpg"))
#         # 如果有其他格式，可以扩展，例如：
#         # image_list = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
        
#         total_count = len(image_list)
#         if total_count == 0:
#             print(f"⚠️ {target_set} 下未找到图片，跳过。")
#             continue

#         print(f"📸 找到 {total_count} 张图片，开始批量推理 (Batch Inference)...")

#         # ================= 核心加速修改 =================
#         # batch=16: 一次处理 16 张图 (根据显存大小调整，显存大可调至 32, 64 或 -1 自动)
#         # save=False: 不使用 YOLO 默认保存路径，我们手动保存以符合你的目录结构
#         # verbose=False: 关闭 YOLO 内部打印
#         # imgsz: 确保推理尺寸一致 (可选，默认模型训练尺寸)
#         results = model.predict(
#             source=image_list, 
#             batch=1,          # 【关键】批处理大小，显著加速
#             save=False,        # 【关键】关闭自动保存，使用下方手动保存
#             verbose=True, 
#             conf=0.25,         # 根据你的需求调整置信度
#             iou=0.5,           # 根据你的需求调整 NMS
#             imgsz=640,
#             stream=True        
#         )
#         # ==============================================

#         # 遍历结果并保存 (保持你原有的路径和文件生成逻辑)
#         # results 是一个列表，顺序与 image_list 一致
#         for result in tqdm(results, desc=f"Saving {target_set}", unit="img"):
#             original_path = Path(result.path)
            
#             try:
#                 rel_path = original_path.relative_to(input_dir)
#             except ValueError:
#                 rel_path = Path(original_path.name)
                
#             new_path = output_base / rel_path
#             new_path.parent.mkdir(parents=True, exist_ok=True)
            
#             # 1. 保存结果图片 (手动调用 save，指定文件名)
#             # 注意：result.save() 内部会重新推理一次如果 boxes 没生成，但这里 boxes 已存在
#             # 为了最快，建议直接用 cv2 绘制或者 result.save(filename=...)
#             result.save(filename=str(new_path))

#             # 2. 保存 .txt 标签
#             txt_path = new_path.with_suffix('.txt')
#             result.save_txt(str(txt_path))

#             # 3. 保存 .json 结果
#             json_path = new_path.with_suffix('.json')
#             json_str = result.to_json()
#             parsed_json = json.loads(json_str)

#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(parsed_json, f, ensure_ascii=False, indent=2)

#         print(f"\n✅ {target_set} 集处理完成！")

# if __name__ == '__main__':
#     run_test()


from ultralytics import YOLO
from pathlib import Path
import json
import os
from tqdm import tqdm

# ===================== 配置参数 =====================

MODEL_PATH = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/cellseg/260323_MAIN_yolo11m/weights/best.pt"

DATA_ROOTS = {
    "test_FXH_noALL": Path("/root/autodl-tmp/data/FXH_imgs_noALL_260318"),
    "test_BJH": Path("/root/autodl-tmp/data/BJH_imgs_260211"),
    "test_TJMU": Path("/root/autodl-tmp/data/TJMU_imgs_260318"),
    "train": Path("/root/autodl-tmp/data/MAIN_imgs_split_260323/Train"),
    "val": Path("/root/autodl-tmp/data/MAIN_imgs_split_260323/Val"),
}

PRED_ROOT = Path("/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolo_preds_260323/")

target_sets = [ "test_FXH_noALL",
               "test_BJH",
               "test_TJMU",
               "train",
               "val"]

# target_set = "val"

# ===================== 执行推理 =====================

def run_test():
    for target_set in target_sets:
        input_dir = DATA_ROOTS[target_set]
        output_base = PRED_ROOT / target_set

        # 加载模型
        model = YOLO(MODEL_PATH)

        print(f"🔍 正在扫描 {target_set} 目录下的图片...")
        image_list = list(input_dir.rglob("*.jpg"))
        total_count = len(image_list)
        print(f"📸 找到 {total_count} 张图片，准备开始逐一推理...")

        # 使用 tqdm 直接包装图片路径列表
        pbar = tqdm(image_list, desc=f"Testing {target_set}", unit="img")

        for img_path in pbar:
            # 核心修改：每次只传入一张图片的路径字符串
            # verbose=False 可以关闭 YOLO 默认的单行打印，让进度条更干净
            results = model(str(img_path), verbose=False) 
            result = results[0] # 因为只传了一张图，所以取第一个结果
            
            # --- 严格保持你原本的路径生成和命名逻辑 ---
            original_path = Path(result.path)
            
            try:
                rel_path = original_path.relative_to(input_dir)
            except ValueError:
                rel_path = Path(original_path.name)
                
            new_path = output_base / rel_path
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存结果图片
            result.save(filename=str(new_path))

            # 2. 保存 .txt 标签
            txt_path = new_path.with_suffix('.txt')
            result.save_txt(str(txt_path))

            # 3. 保存 .json 结果
            json_path = new_path.with_suffix('.json')
            json_str = result.to_json()
            parsed_json = json.loads(json_str)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            
            # 在进度条右侧显示当前文件名
            pbar.set_postfix({"file": original_path.name})

        print(f"\n✅ {target_set} 集处理完成！")

if __name__ == '__main__':
    run_test()