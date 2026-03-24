#无可解释性 有/无预训练 13版  datamodule.py 完整版（已改成 Undersample 负样本到 1:2） 可运行
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import torch
from .datasets import LabelFileDataset
import pytorch_lightning as pl
import random

class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, cfg_data, batch_size, num_workers, pin_memory=True):
        super().__init__()
        self.cfg = cfg_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds = None
        self.val_ds = None
        self.class_weights = None
        self.train_sampler = None

    def setup(self, stage=None):
        self.train_ds = LabelFileDataset(
            self.cfg["train_labels"],
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            self.cfg.get("augment"),
            True,
        )
        self.val_ds = LabelFileDataset(
            self.cfg["val_labels"],
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            None,
            False,
        )

        # class weights（原逻辑保留）
        counts = Counter([y for _, y in self.train_ds.samples])
        total = sum(counts.values())

        #self.class_weights = torch.tensor([1.0, 2.0])
        self.class_weights = torch.tensor(
            [
                0.0 if counts.get(c, 0) == 0 else total / (self.cfg["num_classes"] * counts.get(c, 0))
                for c in range(self.cfg["num_classes"])
            ],
            dtype=torch.float,
        )

        # # Undersample 负样本
        # pos_samples = [s for s in self.train_ds.samples if s[1] == 1]
        # neg_samples = [s for s in self.train_ds.samples if s[1] == 0]

        # print(f"[Data] 原训练集: 正样本 {len(pos_samples)}, 负样本 {len(neg_samples)}")

        # # Undersample 负样本到 正样本的 2 倍（1:2 比例，医学分类最常用）
        # target_neg = len(pos_samples) * 2
        # if len(neg_samples) > target_neg:
        #     random.seed(42)
        #     neg_samples = random.sample(neg_samples, target_neg)
        #     print(f"[Data] Undersample 负样本 → {len(neg_samples)}")
        # else:
        #     print(f"[Data] 负样本不足 2 倍，不 undersample")

        # # 最终训练集 = 全部正样本 + undersample 后的负样本
        # balanced_samples = pos_samples + neg_samples
        # self.train_ds.samples = balanced_samples   # 关键！直接替换 dataset 的 samples

        # 关闭 WeightedRandomSampler（已经手动平衡了）
        self.cfg["use_weighted_sampler"] = False
        self.train_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_exception(self, exception: BaseException):
        pass

'''
# src/datamodule.py 有可解释性
from torch.utils.data import DataLoader
from collections import Counter
import torch
from .datasets import LabelFileDataset
import pytorch_lightning as pl
import os

class SingleCellDataModule(pl.LightningDataModule):
    def __init__(self, cfg_data, batch_size, num_workers, pin_memory=True):
        super().__init__()
        self.cfg = cfg_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        # === 定义特征 CSV 路径 (指向 feature_o/singlecell_o_f) ===
        # 注意：这里我们假设你的目录结构是固定的
        base_feature_dir = "/root/autodl-tmp/projects/mwh/SingleCellProject/feature_o/singlecell_o_f"
        train_csv = os.path.join(base_feature_dir, "train", "features.csv")
        val_csv   = os.path.join(base_feature_dir, "val", "features.csv")

        # 初始化训练集
        self.train_ds = LabelFileDataset(
            self.cfg["train_labels"], # 来自 data.yaml
            train_csv,                # 传入 CSV
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            self.cfg.get("augment"),
            True,
        )
        
        # 初始化验证集
        self.val_ds = LabelFileDataset(
            self.cfg["val_labels"],   # 来自 data.yaml
            val_csv,                  # 传入 CSV
            self.cfg["img_size"],
            self.cfg["mean"],
            self.cfg["std"],
            None,
            False,
        )

        # 打印数据统计
        pos_samples = [s for s in self.train_ds.samples if s[1] == 1]
        neg_samples = [s for s in self.train_ds.samples if s[1] == 0]
        print(f"[Data] Train set setup complete.")
        print(f"       Positive: {len(pos_samples)}")
        print(f"       Negative: {len(neg_samples)}")
        print(f"       Ratio 1:{len(neg_samples)/len(pos_samples):.2f}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True, # 训练集打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True # 保持 worker 活跃加速训练
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
'''