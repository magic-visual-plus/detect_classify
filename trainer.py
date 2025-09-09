import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from model.DINOv3 import DinoV3Classifier
from torchmetrics.classification import Accuracy, Precision, Recall, ConfusionMatrix
import utils
import utils.dataset
import utils.dataloader
import utils.transforms
from config import load_config, parse_args, args_to_config_dict


class DinoV3ClassifierTrainer(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.num_classes = config["num_classes"]
        self.class_names = config.get("class_names", [f"Class_{i}" for i in range(self.num_classes)])
        
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
            
        # 添加混淆矩阵用于计算每个类别的准确率
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.train_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        
    def training_step(self, batch):
        """
        training_step is called at the end of each training epoch.
        """
        images, labels, info = batch
        outputs = self.model(images)
        
        if self.num_classes == 2:
            labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
            loss = self.loss_fn(outputs, labels_float)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # [batch_size]
        else:
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            # 更新训练混淆矩阵
            self.train_confusion_matrix(preds, labels)

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
        images, labels, info = batch
        outputs = self.model(images)
        
        # 处理二分类和多分类的不同情况
        if self.num_classes == 2:
            labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
            loss = self.loss_fn(outputs, labels_float)
            # 二分类预测：sigmoid + 阈值0.5
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # [batch_size]
        else:
            # 多分类：outputs shape [batch_size, num_classes], labels shape [batch_size]
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            # 更新验证混淆矩阵
            self.val_confusion_matrix(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy(preds, labels), prog_bar=True)
        self.log("val_precision", self.val_precision(preds, labels), prog_bar=True)
        self.log("val_recall", self.val_recall(preds, labels), prog_bar=True)
        
        return loss

    def calculate_per_class_metrics(self, confusion_matrix):
        """
        计算每个类别的准确率（accuracy）、召回率（recall）、精确率（precision）
        """
        cm = confusion_matrix.compute()
        per_class_accuracy = []
        per_class_recall = []
        per_class_precision = []

        for i in range(self.num_classes):
            tp = cm[i, i].item()
            total = cm[i, :].sum().item()  # 该类别的真实样本数（TP + FN）
            predicted = cm[:, i].sum().item()  # 被预测为该类别的样本数（TP + FP）
            all_samples = cm.sum().item()  # 总样本数

            # 准确率：该类别被正确预测的样本数 / 总样本数
            if all_samples > 0:
                acc = tp / all_samples
            else:
                acc = 0.0
            per_class_accuracy.append(acc)

            # 召回率：TP / (TP + FN)
            if total > 0:
                recall = tp / total
            else:
                recall = 0.0
            per_class_recall.append(recall)

            # 精确率：TP / (TP + FP)
            if predicted > 0:
                precision = tp / predicted
            else:
                precision = 0.0
            per_class_precision.append(precision)

        return per_class_accuracy, per_class_recall, per_class_precision

    
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
        
        # 计算每个类别的准确率、召回率和精确率
        if self.num_classes > 2:
            per_class_acc, per_class_recall, per_class_precision = self.calculate_per_class_metrics(self.val_confusion_matrix)
            
            # 记录每个类别的准确率、召回率和精确率
            for i in range(self.num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                self.log(f"val_acc_{class_name}", per_class_acc[i], prog_bar=False)
                self.log(f"val_recall_{class_name}", per_class_recall[i], prog_bar=False)
                self.log(f"val_precision_{class_name}", per_class_precision[i], prog_bar=False)
            
            
            data = {
                "Class": [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" for i in range(self.num_classes)],
                "Accuracy": [per_class_acc[i] for i in range(self.num_classes)],
                "Recall": [per_class_recall[i] for i in range(self.num_classes)],
                "Precision": [per_class_precision[i] for i in range(self.num_classes)],
            }
            df = pd.DataFrame(data)
            print("\n" + "="*50)
            print("Per-class validation metrics:")
            print("="*50)
            print(df.to_string(index=False, float_format="%.4f", col_space=12))
            print("="*50)
        
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        if self.num_classes > 2:
            self.val_confusion_matrix.reset()
    
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

def get_model(config_dict):
    backbone_weights = config_dict.get('backbone_weights', None)
    if backbone_weights is None:
        backbone_name = config_dict['backbone_name']
    else:
        name_split_len = 2 if 'vit' in backbone_weights else 3
        backbone_name = "_".join(backbone_weights.split("/")[-1].split("_")[:name_split_len])
    print(f"backbone name: {backbone_name}")
    print(f"backbone weights: {backbone_weights}")
    return DinoV3Classifier(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        num_classes=config_dict['num_classes'],
        check_point_path=config_dict.get('check_point_path', None),
        REPO_DIR=config_dict['REPO_DIR'],
        freeze_backbone=True
    )

def main(config_dict):
    """
    主训练函数
    """
    backbone_weights = config_dict['backbone_weights']
    name_split_len = 2 if 'vit' in backbone_weights else 3
    backbone_name = "_".join(backbone_weights.split("/")[-1].split("_")[:name_split_len])
    print(f"backbone name: {backbone_name}")
    print(f"backbone weights: {backbone_weights}")

    # 对coco_file 进行前处理
    def post_coco_file(coco_file, new_coco_file=None) -> dict:
        coco_data = utils.dataset.load_coco_file(coco_file)
        coco_data = utils.dataset.filter_and_remap_coco_categories(
            coco_data, 
            config_dict['target_category_names']
        )
        # coco_data = utils.dataset.filter_coco_by_source_category(
        #     coco_data,
        #     'gt'
        # )
        coco_data = utils.dataset.correct_predicted_categories(
            coco_data, 
            iou_threshold=config_dict['iou_threshold']
        )
        coco_data = utils.dataset.merge_coco_categories(
            coco_data,
            merge_dict=config_dict['merge_coco_dict']
        )

        if new_coco_file:
            utils.dataset.save_coco_file(coco_data, new_coco_file)
        return coco_data

    train_coco_ann = config_dict['dataset']['train_ann'] + ".tmp"
    valid_coco_ann = config_dict['dataset']['val_ann'] + ".tmp"
    try:
        os.remove(train_coco_ann)
    except FileNotFoundError:
        pass
    try:
        os.remove(valid_coco_ann)
    except FileNotFoundError:
        pass
    train_coco_data = post_coco_file(config_dict['dataset']['train_ann'], train_coco_ann)
    valid_coco_data = post_coco_file(config_dict['dataset']['val_ann'], valid_coco_ann)

    # print(train_coco_data)
    # print(valid_coco_data)

    train_loader, train_dataset, val_loader, val_dataset = utils.dataloader.create_train_val_dataloaders(
        train_root=config_dict['dataset']['train_path'],
        train_ann=train_coco_ann,
        val_root=config_dict['dataset']['val_path'],
        val_ann=valid_coco_ann,
        train_transform=utils.transforms.get_default_transform(size=(config_dict['resize']['width'], config_dict['resize']['height'])),
        val_transform=utils.transforms.get_default_transform(is_train=False, size=(config_dict['resize']['width'], config_dict['resize']['height'])),
        min_bbox_area=config_dict['min_bbox_area'],
        batch_size=config_dict['batch_size'],
        num_workers=config_dict['num_workers'],
    )
    # 打印不同类别样本数量（训练集和验证集）
    from collections import Counter

    def print_category_sample_counts(dataset, dataset_name="train"):
        cat_id_list = []
        for ann in dataset.annotations:
            cat_id_list.append(ann['category_id'])
        counter = Counter(cat_id_list)
        print(f"\n{dataset_name} category sample counts:")
        for idx, name in enumerate(dataset.cat_names):
            print(f"  {name}: {counter.get(idx, 0)}")

    print_category_sample_counts(train_dataset, "Train")
    print_category_sample_counts(val_dataset, "Val")

    print(train_dataset.cat_names)
    num_classes = len(train_dataset.cat_names)
    print(f"num_classes: {num_classes}")
    config_dict['num_classes'] = num_classes
    config_dict['class_names'] = train_dataset.cat_names
    dinov3_classifier = DinoV3Classifier(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        num_classes=num_classes,
        freeze_backbone=True
    )
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

if __name__ == "__main__":
    args = parse_args()
    
    config_manager = load_config(args.config)
    
    override_config = args_to_config_dict(args)
    if override_config:
        config_manager.merge_config(override_config)
    
    config_dict = config_manager.to_dict()
    
    print("=" * 50)
    print("Train Config:")
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