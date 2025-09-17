import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import lightning as L
from model.DINOv3 import DinoV3Classifier
from torchmetrics.classification import Accuracy, Precision, Recall
import utils
import utils.dataloader
import utils.transforms
from config import load_config, parse_args, args_to_config_dict


class DinoV3ClassifierTrainer(L.LightningModule):
    def __init__(self, model, config):
        """
        __init__ is called at the beginning of the training.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.num_classes = config["num_classes"]
        
        # 根据类别数量选择合适的损失函数和指标
        if self.num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.train_accuracy = Accuracy(task="binary")
            self.train_precision = Precision(task="binary", average='macro')
            self.train_recall = Recall(task="binary", average='macro')
            self.val_accuracy = Accuracy(task="binary")
            self.val_precision = Precision(task="binary", average='macro')
            self.val_recall = Recall(task="binary", average='macro')
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.train_precision = Precision(task="multiclass", num_classes=self.num_classes, average='macro')
            self.train_recall = Recall(task="multiclass", num_classes=self.num_classes, average='macro')
            self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_precision = Precision(task="multiclass", num_classes=self.num_classes, average='macro')
            self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average='macro')
    
    def training_step(self, batch):
        """
        training_step is called at the end of each training epoch.
        """
        images, labels = batch
        outputs = self.model(images)
        
        if self.num_classes == 2:
            # 二分类：outputs 是 [batch_size, 1]，labels 是 [batch_size]
            labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
            loss = self.loss_fn(outputs, labels_float)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # [batch_size]
        else:
            # 多分类：outputs 是 [batch_size, num_classes]，labels 是 [batch_size]
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy(preds, labels), prog_bar=True)
        self.log("train_precision", self.train_precision(preds, labels), prog_bar=True)
        self.log("train_recall", self.train_recall(preds, labels), prog_bar=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer and self.trainer.optimizers else None
        if current_lr is not None:
            self.log("lr", current_lr, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        """
        validation_step is called at the end of each validation epoch.
        """
        images, labels = batch
        outputs = self.model(images)
        
        if self.num_classes == 2:
            # 二分类：outputs 是 [batch_size, 1]，labels 是 [batch_size]
            labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
            loss = self.loss_fn(outputs, labels_float)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # [batch_size]
        else:
            # 多分类：outputs 是 [batch_size, num_classes]，labels 是 [batch_size]
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy(preds, labels), prog_bar=True)
        self.log("val_precision", self.val_precision(preds, labels), prog_bar=True)
        self.log("val_recall", self.val_recall(preds, labels), prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """
        on_validation_epoch_end is called at the end of each validation epoch.
        """
        val_acc = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        
        self.log("val_acc_epoch", val_acc, prog_bar=False)
        self.log("val_precision_epoch", val_precision, prog_bar=False)
        self.log("val_recall_epoch", val_recall, prog_bar=False)
        
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
    
    def configure_optimizers(self):
        """
        configure_optimizers is called at the beginning of the training.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["max_epochs"])
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.config["max_epochs"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
class ClassificationData(L.LightningDataModule):
    def __init__(
            self, 
            train_dataloader,
            val_dataloader,
            batch_size=256, 
            num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        return self._val_dataloader

def main(config_dict):
    """
    主训练函数
    
    Args:
        config_dict: 配置字典
    """
    backbone_weights = config_dict['backbone_weights']
    name_split_len = 2 if 'vit' in backbone_weights else 3
    backbone_name = "_".join(backbone_weights.split("/")[-1].split("_")[:name_split_len])
    print(f"backbone name: {backbone_name}")
    print(f"backbone weights: {backbone_weights}")

    # 检查dataset配置
    dataset_config = config_dict['dataset']
    
    if isinstance(dataset_config, dict) and 'train' in dataset_config and 'val' in dataset_config:
        # 如果提供了训练集和验证集路径，直接使用
        print("使用提供的训练集和验证集路径")
        train_dataset = EMDS6Dataset(root_dir=dataset_config['train'])
        val_dataset = EMDS6Dataset(root_dir=dataset_config['val'])
        
        # 设置transform
        train_dataset.transform = utils.transforms.get_default_transform()
        val_dataset.transform = utils.transforms.get_default_transform(is_train=False)
        
    else:
        # 如果只提供了一个路径，使用random_split进行划分
        print("使用单个数据集路径，自动划分训练集和验证集")
        from torch.utils.data import random_split
        
        dataset = EMDS6Dataset(root_dir=dataset_config)
        
        val_ratio = config_dict.get('val_ratio', 0.2)
        total_size = len(dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 分别为训练集和验证集设置不同的transform
        train_dataset.dataset.transform = utils.transforms.get_default_transform()
        val_dataset.dataset.transform = utils.transforms.get_default_transform(is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config_dict['batch_size'],
        shuffle=True,
        num_workers=config_dict['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        num_workers=config_dict['num_workers']
    )
    # 动态获取实际类别数量
    actual_num_classes = len(train_dataset.classes)
    print(f"实际类别数量: {actual_num_classes}")
    print(f"类别列表: {train_dataset.classes}")
    
    # 验证训练集和验证集的类别是否一致
    if isinstance(dataset_config, dict) and 'train' in dataset_config and 'val' in dataset_config:
        if train_dataset.classes != val_dataset.classes:
            print("警告：训练集和验证集的类别不一致！")
            print(f"训练集类别: {train_dataset.classes}")
            print(f"验证集类别: {val_dataset.classes}")
    
    dinov3_classifier = DinoV3Classifier(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        num_classes=actual_num_classes,  # 使用实际类别数量
        freeze_backbone=True
    )
    # 更新配置中的类别数量
    config_dict["num_classes"] = actual_num_classes
    model = DinoV3ClassifierTrainer(dinov3_classifier, config_dict)
    print(model.model)
    data = ClassificationData(
        train_loader,
        val_loader,
        batch_size=config_dict["batch_size"], 
        num_workers=config_dict["num_workers"], 
    )
    
    trainer = L.Trainer(
        max_epochs=config_dict["max_epochs"], 
        accelerator="gpu", 
        devices=1,
        log_every_n_steps=1,     
        enable_progress_bar=True,  
        enable_model_summary=True, 
    )
    
    trainer.fit(model, data)


class EMDS6Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # 获取类别目录
        self.images = []
        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.images.append((os.path.join(cls_path, img_name), idx))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
if __name__ == "__main__":
    config_dict = {
        'lr': 0.0001, 
        'max_epochs': 100, 
        'batch_size': 64, 
        'num_workers': 12, 
        'num_classes': 2, 
        # 方式1：使用单个数据集路径，自动划分训练集和验证集
        # 'dataset': "/root/褶皱样本",
        
        # 方式2：提供训练集和验证集路径，不进行自动划分
        'dataset': {
            'train': "/root/褶皱样本train",
            'val': "/root/褶皱样本valid"
        },
        
        'min_bbox_area': 0, 
        'backbone_weights': '/root/autodl-tmp/seat_model/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    }
    
    print("=" * 50)
    print("训练配置:")
    print("=" * 50)
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value} ({type(sub_value)})")
        else:
            print(f"{key}: {value} ({type(value)})")
    print("=" * 50)

    main(config_dict)