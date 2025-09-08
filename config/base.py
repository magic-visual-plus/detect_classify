"""
基础配置类
提供默认配置和配置验证
"""

from typing import Dict, Any, Optional
from .config_manager import ConfigManager


class BaseConfig:
    """基础配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "max_epochs": 30,
        "batch_size": 64,
        "num_workers": 12,
        "dataset": {
            "train_path": "/root/autodl-tmp/seat_dataset/chengdu_customer",
            "val_path": "/root/autodl-tmp/seat_dataset/chengdu_valid/",
            "train_ann": "/root/autodl-tmp/seat_dataset/chengdu_customer/_classification.coco.json",
            "val_ann": "/root/autodl-tmp/seat_dataset/chengdu_valid/_classification.coco.json",
        },
        "min_bbox_area": 400,
        "backbone_weights": "/root/autodl-tmp/seat_model/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化基础配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_manager = ConfigManager()
        
        if config_path:
            self.config_manager.load_from_yaml(config_path)
        else:
            self.config_manager.load_from_dict(self.DEFAULT_CONFIG)
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            配置字典
        """
        return self.config_manager.to_dict()
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        required_keys = [
            'lr', 'max_epochs', 'batch_size', 'num_workers',
            'dataset', 'min_bbox_area', 'backbone_weights'
        ]
        
        for key in required_keys:
            if not self.config_manager.get(key):
                raise ValueError(f"缺少必需的配置项: {key}")
        
        # 验证数据集配置
        dataset_keys = ['train_path', 'val_path', 'train_ann', 'val_ann']
        for key in dataset_keys:
            if not self.config_manager.get(f'dataset.{key}'):
                raise ValueError(f"缺少必需的数据集配置项: dataset.{key}")
        
        return True
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config_manager[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.config_manager[key] = value


# 为了向后兼容，保留原有的 _base_ 变量
_base_ = BaseConfig.DEFAULT_CONFIG

