import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
from torchvision import transforms


class COCOClassificationDataset(Dataset):
    """
    COCO数据集的分类任务Dataset类
    将每个标注框裁剪出来作为一个独立的分类样本
    """
    
    def __init__(self, 
                 root_dir, 
                 ann_file, 
                 transform=None,
                 min_bbox_area=400,
                 target_categories=None):
        """
        Args:
            root_dir (str): COCO图片根目录路径
            ann_file (str): COCO标注文件路径 (json格式)
            transform: 图像变换/数据增强
            min_bbox_area (float): 最小边界框面积，过滤掉太小的目标
            target_categories (list): 指定要使用的类别名称列表，None表示使用所有类别
        """
        self.root_dir = root_dir
        self.transform = transform
        self.min_bbox_area = min_bbox_area
        
        # 加载COCO标注
        self.coco = COCO(ann_file)
        
        # 获取类别信息
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_names = [cat['name'] for cat in self.categories]
        
        # 如果指定了目标类别，进行过滤
        if target_categories:
            self.cat_ids = [cat['id'] for cat in self.categories 
                           if cat['name'] in target_categories]
        else:
            self.cat_ids = [cat['id'] for cat in self.categories]
        
        # 创建类别ID到索引的映射（用于分类任务）
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.label_to_cat_name = {idx: self.coco.loadCats([cat_id])[0]['name'] 
                                 for cat_id, idx in self.cat_id_to_label.items()}
        
        # 获取所有有效的标注（annotation）
        self.annotations = []
        for cat_id in self.cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=[cat_id])
            anns = self.coco.loadAnns(ann_ids)
            
            # 过滤掉太小的边界框
            for ann in anns:
                if ann['area'] >= self.min_bbox_area:
                    self.annotations.append(ann)
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        返回裁剪后的图片和对应的类别标签
        """
        # 获取标注信息
        ann = self.annotations[idx]
        
        # 获取图片信息
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        
        bbox = ann['bbox']
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        img_width, img_height = image.size
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_width, x + w)
        y2 = min(img_height, y + h)
        
        cropped_image = image.crop((x, y, x2, y2))
        
        category_id = ann['category_id']
        label = self.cat_id_to_label[category_id]
        
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        info = {
            'image_id': ann['image_id'],
            'annotation_id': ann['id'],
            'category_name': self.label_to_cat_name[label],
            'bbox': bbox,
            'original_size': (img_width, img_height),
            'image_name': img_info['file_name'],
            'annotation_source': ann['source']
        }
        
        return cropped_image, label, info
    
    def get_num_classes(self):
        """返回类别数量"""
        return len(self.cat_ids)
    
    def get_class_names(self):
        """返回类别名称列表"""
        return [self.label_to_cat_name[i] for i in range(len(self.cat_ids))]


if __name__ == "__main__":
    # 配置路径
    train_root = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/train"
    train_ann = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/train/_annotations.coco.json"
    
    val_root = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/valid"
    val_ann = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/valid/_annotations.coco.json"