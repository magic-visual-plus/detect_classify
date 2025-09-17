import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
from utils.dataset import COCOClassificationDataset
from PIL import Image


def create_coco_dataloader(root_dir, 
                          ann_file, 
                          transform,
                          batch_size=32,
                          min_bbox_area=40,
                          crop_scale_factor=1.0,
                          is_train=True,
                          target_categories=None,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=None,
                          pad_mode="constant",
                          pad_color=(114, 114, 114)):
    """
    创建COCO分类任务的DataLoader
    
    Args:
        root_dir: COCO图片目录
        ann_file: 标注文件路径
        transform: 数据增强方式
        batch_size: 批次大小
        min_bbox_area: 
        is_train: 是否为训练集
        target_categories: 指定的类别列表
        num_workers: 数据加载的工作进程数
        pin_memory: 是否使用pin_memory
        shuffle: 是否打乱数据, None时自动根据is_train决定
        pad_mode: 越界填充方式 ("constant", "edge", None)
        pad_color: 常数填充的颜色 (R, G, B)
    """
    if shuffle is None:
        shuffle = is_train
    
    dataset = COCOClassificationDataset(
        root_dir=root_dir,
        ann_file=ann_file,
        transform=transform,
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        target_categories=target_categories,
        pad_mode=pad_mode,
        pad_color=pad_color
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train  # 训练时丢弃不完整的批次
    )
    
    return dataloader, dataset


def create_distributed_dataloader(root_dir,
                                ann_file,
                                transform,
                                batch_size=32,
                                min_bbox_area=400,
                                crop_scale_factor=1.0,
                                is_train=True,
                                target_categories=None,
                                num_workers=4,
                                pin_memory=True,
                                world_size=None,
                                rank=None):
    """
    创建分布式训练的DataLoader
    
    Args:
        world_size: 总进程数
        rank: 当前进程的rank
    """
    
    dataset = COCOClassificationDataset(
        root_dir=root_dir,
        ann_file=ann_file,
        transform=transform,
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        target_categories=target_categories
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=is_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train
    )
    
    return dataloader, dataset, sampler



def get_dataloader_stats(dataloader, dataset):
    """
    获取数据加载器的统计信息
    
    Returns:
        dict: 包含数据集统计信息的字典
    """
    stats = {
        'dataset_size': len(dataset),
        'num_classes': dataset.get_num_classes(),
        'class_names': dataset.get_class_names(),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'total_samples': len(dataloader) * dataloader.batch_size
    }
    
    # 计算类别分布
    class_counts = {}
    for _, labels, _ in dataloader:
        for label in labels:
            label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1
    
    stats['class_distribution'] = class_counts
    
    return stats


def create_train_val_dataloaders(train_root,
                                train_ann,
                                train_transform,
                                val_root,
                                val_ann,
                                val_transform,
                                batch_size=32,
                                min_bbox_area=100,
                                crop_scale_factor=1.0,
                                target_categories=None,
                                num_workers=4,
                                pad_mode="constant",
                                pad_color=(114, 114, 114)):
    """
    同时创建训练和验证数据加载器
    """
    train_loader, train_dataset = create_coco_dataloader(
        root_dir=train_root,
        ann_file=train_ann,
        transform=train_transform,
        batch_size=batch_size,
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        is_train=True,
        target_categories=target_categories,
        num_workers=num_workers,
        pad_mode=pad_mode,
        pad_color=pad_color
    )
    
    val_loader, val_dataset = create_coco_dataloader(
        root_dir=val_root,
        ann_file=val_ann,
        transform=val_transform,
        batch_size=batch_size,
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        is_train=False,
        target_categories=target_categories,
        num_workers=num_workers,
        pad_mode=pad_mode,
        pad_color=pad_color
    )
    
    return train_loader, train_dataset, val_loader, val_dataset


# 数据预处理工具函数
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图像张量，用于可视化
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized


def tensor_to_pil(tensor):
    """
    将张量转换为PIL图像
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一个批次
    
    # 反归一化
    tensor = denormalize_image(tensor)
    
    # 转换为PIL图像
    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    tensor = (tensor * 255).byte().cpu().numpy()
    
    return Image.fromarray(tensor)


def get_batch_info(batch):
    """
    获取批次信息
    
    Args:
        batch: (images, labels, info) 元组
    
    Returns:
        dict: 批次信息
    """
    images, labels, info = batch
    
    batch_info = {
        'images_shape': images.shape,
        'labels_shape': labels.shape,
        'batch_size': images.shape[0],
        'num_classes': len(torch.unique(labels)),
        'class_distribution': torch.bincount(labels).tolist(),
        'sample_info': {
            'image_ids': info['image_id'][:5],  # 前5个样本的ID
            'category_names': info['category_name'][:5]  # 前5个样本的类别
        }
    }
    
    return batch_info


def create_merged_dataloaders(train_root,
                             train_ann,
                             val_root,
                             val_ann,
                             train_transform,
                             val_transform,
                             batch_size=32,
                             min_bbox_area=100,
                             crop_scale_factor=1.0,
                             target_categories=None,
                             num_workers=4,
                             pad_mode="constant",
                             pad_color=(114, 114, 114),
                             train_ratio=0.8,
                             random_seed=42):
    """
    合并训练集和验证集，然后重新划分数据集
    
    Args:
        train_root: 训练集图片目录
        train_ann: 训练集标注文件路径
        val_root: 验证集图片目录
        val_ann: 验证集标注文件路径
        train_transform: 训练时的数据增强
        val_transform: 验证时的数据增强
        batch_size: 批次大小
        min_bbox_area: 最小边界框面积
        crop_scale_factor: 裁剪缩放因子
        target_categories: 指定的类别列表
        num_workers: 数据加载的工作进程数
        pad_mode: 越界填充方式
        pad_color: 常数填充的颜色
        train_ratio: 训练集比例 (0-1之间)
        random_seed: 随机种子
    
    Returns:
        tuple: (train_loader, train_dataset, val_loader, val_dataset, merged_dataset)
    """
    # 设置随机种子确保可重现性
    torch.manual_seed(random_seed)
    
    # 创建原始训练集和验证集
    train_dataset = COCOClassificationDataset(
        root_dir=train_root,
        ann_file=train_ann,
        transform=None,  # 先不应用transform，合并后再应用
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        target_categories=target_categories,
        pad_mode=pad_mode,
        pad_color=pad_color
    )
    
    val_dataset = COCOClassificationDataset(
        root_dir=val_root,
        ann_file=val_ann,
        transform=None,  # 先不应用transform，合并后再应用
        min_bbox_area=min_bbox_area,
        crop_scale_factor=crop_scale_factor,
        target_categories=target_categories,
        pad_mode=pad_mode,
        pad_color=pad_color
    )
    
    # 合并数据集
    merged_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # 计算划分大小
    total_size = len(merged_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # 重新划分数据集
    train_indices, val_indices = random_split(
        range(total_size), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 创建新的训练和验证数据集
    class MergedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, transform, train_dataset, val_dataset):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = transform
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
            # 合并annotations（用于统计，但实际访问通过__getitem__）
            self.all_annotations = train_dataset.annotations + val_dataset.annotations
            
            # 复制其他必要属性
            self.cat_names = train_dataset.cat_names
            self.cat_ids = train_dataset.cat_ids
            self.cat_id_to_label = train_dataset.cat_id_to_label
            self.label_to_cat_name = train_dataset.label_to_cat_name
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            # 获取原始索引
            original_idx = self.indices[idx]
            
            # 找到在哪个子数据集中
            if original_idx < len(self.train_dataset):
                # 在训练集中
                item = self.train_dataset[original_idx]
            else:
                # 在验证集中
                val_idx = original_idx - len(self.train_dataset)
                item = self.val_dataset[val_idx]
            
            # 应用transform
            if self.transform is not None:
                image, label, info = item
                image = self.transform(image)
                return image, label, info
            
            return item
        
        def get_num_classes(self):
            return self.train_dataset.get_num_classes()
        
        def get_class_names(self):
            return self.train_dataset.get_class_names()
        
        @property
        def annotations(self):
            """返回重新划分后的annotations"""
            return [self.all_annotations[i] for i in self.indices]
    
    # 创建新的训练和验证数据集
    new_train_dataset = MergedDataset(merged_dataset, train_indices, train_transform, train_dataset, val_dataset)
    new_val_dataset = MergedDataset(merged_dataset, val_indices, val_transform, train_dataset, val_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        new_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        new_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, new_train_dataset, val_loader, new_val_dataset


