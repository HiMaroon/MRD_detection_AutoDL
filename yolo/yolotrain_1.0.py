from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型权重文件（.pt）
    
    model = YOLO("/root/autodl-tmp/projects/myq/SingleCellProject/weights/yolo11m-seg.pt")

    # 2. 通过 cfg 参数传入你的训练配置文件

    train_cfg = "/root/autodl-tmp/projects/myq/SingleCellProject/yolo/yolotrain_1.0.yaml"

    # 开始训练
    model.train(cfg=train_cfg,
                batch=32,
                name="wihtout0_MAIN_yolo11m",
                epochs=250,
                patience=50)
    
    # model = YOLO("/root/autodl-tmp/projects/myq/SingleCellProject/weights/yolo11s-seg.pt")
    # model.train(cfg=train_cfg,
    #             batch=64,
    #             name="260313_MAIN_yolo11s",
    #             epochs=250,
    #             patience=50)
    
    # model = YOLO("/root/autodl-tmp/projects/myq/SingleCellProject/weights/yolo11m-seg.pt")
    # model.train(cfg=train_cfg,
    #             batch=32,
    #             name="260313_MAIN_yolo11m",
    #             epochs=250,
    #             patience=50)

    # model = YOLO("/root/autodl-tmp/projects/myq/SingleCellProject/weights/yolo11l-seg.pt")
    # model.train(cfg=train_cfg,
    #             batch=16,
    #             name="260313_MAIN_yolo11l",
    #             epochs=250,
    #             patience=50)