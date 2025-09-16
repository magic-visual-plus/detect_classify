import os
import sys
import  shutil
import yaml
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt

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
            
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.train_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        
        # 用于存储PR曲线数据的列表
        self.val_predictions = []
        self.val_labels = []
        self.val_probabilities = []
        
        if self.num_classes == 2:
            self.val_f1 = 2 * self.val_precision * self.val_recall / (self.val_precision + self.val_recall + 1e-8)
        else:
            self.val_f1 = 2 * self.val_precision * self.val_recall / (self.val_precision + self.val_recall + 1e-8)
        
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
        
        if self.num_classes == 2:
            labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
            loss = self.loss_fn(outputs, labels_float)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # [batch_size]
            
            # 存储二分类的概率输出和标签用于PR曲线计算
            probabilities = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            self.val_probabilities.extend(probabilities)
            self.val_labels.extend(labels.cpu().numpy())
            self.val_predictions.extend(preds.cpu().numpy())
        else:
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            self.val_confusion_matrix(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy(preds, labels), prog_bar=True)
        self.log("val_precision", self.val_precision(preds, labels), prog_bar=True)
        self.log("val_recall", self.val_recall(preds, labels), prog_bar=True)
        
        return loss

    def calculate_per_class_metrics(self, confusion_matrix):
        """
        acc、recall、precision
        """
        cm = confusion_matrix.compute()
        acc = [(cm[i, i].item() / cm.sum().item()) if cm.sum().item() else 0.0 for i in range(self.num_classes)]
        recall = [(cm[i, i].item() / cm[i, :].sum().item()) if cm[i, :].sum().item() else 0.0 for i in range(self.num_classes)]
        precision = [(cm[i, i].item() / cm[:, i].sum().item()) if cm[:, i].sum().item() else 0.0 for i in range(self.num_classes)]
        return acc, recall, precision
    
    def calculate_pr_curve_metrics(self):
        """
        计算PR曲线相关指标
        """
        if self.num_classes != 2 or len(self.val_probabilities) == 0:
            return None
            
        y_true = np.array(self.val_labels)
        y_scores = np.array(self.val_probabilities)
        
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # 计算AUC-PR (Average Precision)
        auc_pr = average_precision_score(y_true, y_scores)
        
        # 计算ROC曲线
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        auc_roc = auc(fpr, tpr)
        
        # 找到最佳阈值（F1分数最大）
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        
        pr_metrics = {
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'precision_curve': precision,
            'recall_curve': recall,
            'thresholds': thresholds
        }
        
        return pr_metrics
    
    def save_pr_curve_plot(self, pr_metrics):
        """
        保存PR曲线和ROC曲线图
        """
        try:
            # 创建保存目录
            save_dir = os.path.join(self.config.get('save_dir'), 'plot')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # PR曲线
            ax1.plot(pr_metrics['recall_curve'], pr_metrics['precision_curve'], 
                    color='blue', linewidth=2, label=f'PR Curve (AUC = {pr_metrics["auc_pr"]:.3f})')
            ax1.axvline(x=pr_metrics['best_recall'], color='red', linestyle='--', 
                       label=f'Best Threshold (F1={pr_metrics["best_f1"]:.3f})')
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
            ax1.set_title('Precision-Recall Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            
            y_true = np.array(self.val_labels)
            y_scores = np.array(self.val_probabilities)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax2.plot(fpr, tpr, color='green', linewidth=2, 
                    label=f'ROC Curve (AUC = {pr_metrics["auc_roc"]:.3f})')
            ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            
            epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
            plot_path = os.path.join(save_dir, f'pr_roc_curves_epoch_{epoch:02d}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"PR曲线和ROC曲线图已保存到: {plot_path}")
            
        except Exception as e:
            print(f"保存PR曲线图时出错: {e}")

    
    def on_validation_epoch_end(self):
        """
        on_validation_epoch_end is called at the end of each validation epoch.
        """
        val_acc = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
        
        self.log("val_acc_epoch", val_acc, prog_bar=False)
        self.log("val_precision_epoch", val_precision, prog_bar=False)
        self.log("val_recall_epoch", val_recall, prog_bar=False)
        self.log("val_f1_epoch", val_f1, prog_bar=False)
        
        # 对于二分类模型，计算PR曲线指标
        if self.num_classes == 2:
            pr_metrics = self.calculate_pr_curve_metrics()
            if pr_metrics is not None:
                # 记录PR曲线指标
                self.log("val_auc_pr", pr_metrics['auc_pr'], prog_bar=False)
                self.log("val_auc_roc", pr_metrics['auc_roc'], prog_bar=False)
                self.log("val_best_f1", pr_metrics['best_f1'], prog_bar=False)
                self.log("val_best_precision", pr_metrics['best_precision'], prog_bar=False)
                self.log("val_best_recall", pr_metrics['best_recall'], prog_bar=False)
                self.log("val_best_threshold", pr_metrics['best_threshold'], prog_bar=False)
                
                # 输出PR曲线指标
                print("\n" + "="*60)
                print("二分类模型 PR曲线指标:")
                print("="*60)
                print(f"AUC-PR (Average Precision): {pr_metrics['auc_pr']:.4f}")
                print(f"AUC-ROC: {pr_metrics['auc_roc']:.4f}")
                print(f"最佳阈值: {pr_metrics['best_threshold']:.4f}")
                print(f"最佳F1分数: {pr_metrics['best_f1']:.4f}")
                print(f"最佳阈值下的Precision: {pr_metrics['best_precision']:.4f}")
                print(f"最佳阈值下的Recall: {pr_metrics['best_recall']:.4f}")
                print("="*60)
                
                # 保存PR曲线图
                self.save_pr_curve_plot(pr_metrics)
        
        if self.num_classes > 2:
            per_class_acc, per_class_recall, per_class_precision = self.calculate_per_class_metrics(self.val_confusion_matrix)
            
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
        
        # 重置指标
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        if self.num_classes > 2:
            self.val_confusion_matrix.reset()
        
        # 清空PR曲线数据
        if self.num_classes == 2:
            self.val_predictions.clear()
            self.val_labels.clear()
            self.val_probabilities.clear()
    
    def configure_optimizers(self):
        """
        configure_optimizers is called at the beginning of the training.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["max_epochs"])
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.config['lr_end_factor'],
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

def post_coco_file(coco_file, new_coco_file, config_dict) -> dict:
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
    train_coco_data = post_coco_file(config_dict['dataset']['train_ann'], train_coco_ann, config_dict)
    valid_coco_data = post_coco_file(config_dict['dataset']['val_ann'], valid_coco_ann, config_dict)

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
        crop_scale_factor=config_dict['crop_scale_factor'],
        batch_size=config_dict['batch_size'],
        num_workers=config_dict['num_workers'],
    )
    # 打印不同类别样本数量（训练集和验证集）
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
    config_dict['freeze_backbone'] = config_dict.get('freeze_backbone', True)
    if config_dict['freeze_backbone'] is False:
        print("freeze_backbone is False, unfreeze the backbone")
        print(type(config_dict['freeze_backbone']))

    dinov3_classifier = DinoV3Classifier(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        num_classes=num_classes,
        freeze_backbone=config_dict['freeze_backbone'],
    )
    model = DinoV3ClassifierTrainer(dinov3_classifier, config_dict)
    print(model.model)
    data = ClassificationData(
        train_loader,
        val_loader,
        batch_size=config_dict["batch_size"], 
        num_workers=config_dict["num_workers"], 
    )

    callbacks = []
    loss_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_loss_model_{epoch:02d}_{val_loss:.4f}",
        save_last=True
    )
    callbacks.append(loss_checkpoint)
    f1_checkpoint = ModelCheckpoint(
        monitor="val_f1_epoch",
        mode="max",
        save_top_k=1,
        filename="best_f1_model_{epoch:02d}_{val_f1_epoch:.4f}",
        save_last=False
    )
    callbacks.append(f1_checkpoint)
    acc_checkpoint = ModelCheckpoint(
        monitor="val_acc_epoch",
        mode="max",
        save_top_k=1,
        filename="best_acc_model_{epoch:02d}_{val_acc_epoch:.4f}",
        save_last=False
    )
    callbacks.append(acc_checkpoint)
    
    trainer = L.Trainer(
        max_epochs=config_dict["max_epochs"], 
        accelerator="gpu", 
        devices=1,
        log_every_n_steps=1,     
        enable_progress_bar=True,  
        enable_model_summary=True, 
        callbacks=callbacks,
        default_root_dir=config_dict.get("save_dir", None),
    )
    if args.save_dir is not None:
        save_dir = config_dict.get('save_dir')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        shutil.copy(train_coco_ann, os.path.join(save_dir, '_annclassfication.train.coco.json'))
        shutil.copy(valid_coco_ann, os.path.join(save_dir,'_annclassfication.valid.coco.json'))
        config_path = os.path.join(args.save_dir, "train_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)
    
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
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)
    
    main(config_dict)