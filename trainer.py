import sys
import torch
import torch.nn as nn
import lightning as L
from model.DINOv3 import DinoV3Classifier
from torchmetrics.classification import Accuracy, Precision, Recall
import utils
import utils.dataloader
import utils.transforms


class DinoV3ClassifierTrainer(L.LightningModule):
    def __init__(self, model, config):
        """
        __init__ is called at the beginning of the training.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.num_classes = config["num_classes"]
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
            num_workers=5,
            dataset_path="."):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.dataset_path = dataset_path

    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        return self._val_dataloader
    

if __name__ == "__main__":
    config = {
        "lr": 1e-4,
        "max_epochs": 100,
        "batch_size": 256, 
        "num_workers": 12,
        "num_classes": 2,
        "dataset":
         {
             "train_path": "/root/autodl-tmp/seat_dataset/chengdu_customer",
             "val_path": "/root/autodl-tmp/seat_dataset/chengdu_valid/",
             "train_ann": "/root/autodl-tmp/seat_dataset/chengdu_customer/_classification.coco.json",
             "val_ann": "/root/autodl-tmp/seat_dataset/chengdu_valid/_classification.coco.json",
         },
         'min_bbox_area': 400,

    }
    backbone_weights = sys.argv[1]
    name_split_len = 2 if 'vit' in backbone_weights else 3
    backbone_name = "_".join(backbone_weights.split("/")[-1].split("_")[:name_split_len])
    print(f"backbone name: {backbone_name}")
    print(f"backbone weights: {backbone_weights}")

    train_loader, train_dataset, val_loader, val_dataset = utils.dataloader.create_train_val_dataloaders(
        train_root=config['dataset']['train_path'],
        train_ann=config['dataset']['train_ann'],
        val_root=config['dataset']['val_path'],
        val_ann=config['dataset']['val_ann'],
        train_transform=utils.transforms.get_default_transform(),
        val_transform=utils.transforms.get_default_transform(is_train=False),
        min_bbox_area=config['min_bbox_area'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    dinov3_classifier = DinoV3Classifier(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        num_classes=config["num_classes"],
        head_embed_dim=1000,
        freeze_backbone=True
    )
    model = DinoV3ClassifierTrainer(dinov3_classifier, config)
    print(model.model)
    data = ClassificationData(
        train_loader,
        val_loader,
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"], 
        train_transform=config["train_transform"],
        val_transform=config["val_transform"]
    )
    
    trainer = L.Trainer(
        max_epochs=config["max_epochs"], 
        accelerator="gpu", 
        devices=1,
        log_every_n_steps=1,     
        enable_progress_bar=True,  
        enable_model_summary=True, 
    )
    
    trainer.fit(model, data)