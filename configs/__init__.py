# E:\SingleCellProject\configs\__init__.py
import os
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

def _load(name: str):
    """读取同目录下的 yaml 并注册到 Hydra ConfigStore，同时返回 dict 供 import 使用"""
    path = os.path.join(os.path.dirname(__file__), f"{name}.yaml")
    cfg = OmegaConf.load(path)
    cs.store(name=name, node=cfg)
    return OmegaConf.to_container(cfg, resolve=True)

# 导出变量供 train.py 直接 import
data_cfg  = _load("data")
model_cfg = _load("model")
train_cfg = _load("train")
wandb_cfg = _load("wandb")
# yolo_cfg  = _load("yolo_singlecell")