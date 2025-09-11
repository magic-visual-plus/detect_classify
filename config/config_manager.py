"""
配置管理器
支持YAML文件加载、命令行参数解析和配置合并
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置字典
        """
        self.config = config_dict.copy()
        return self.config
    
    def merge_config(self, override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置（覆盖现有配置）
        
        Args:
            override_config: 要合并的配置
            
        Returns:
            合并后的配置
        """
        self._deep_update(self.config, override_config)
        return self.config
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        深度更新字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，支持嵌套如 'dataset.train_path'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，支持嵌套如 'dataset.train_path'
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_to_yaml(self, output_path: str) -> None:
        """
        保存配置到YAML文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            配置字典的副本
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.get(key) is not None


def load_config(config_path: str, override_args: Optional[Dict[str, Any]] = None) -> ConfigManager:
    """
    加载配置的便捷函数
    
    Args:
        config_path: 配置文件路径
        override_args: 要覆盖的参数
        
    Returns:
        配置管理器实例
    """
    config_manager = ConfigManager(config_path)
    
    if override_args:
        config_manager.merge_config(override_args)
    
    return config_manager


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models')
    parser.add_argument('--config', '-c', type=str, required=True, 
                       help='Path to config file (YAML format)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of training epochs')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of dataloader worker processes')
    parser.add_argument('--train_path', type=str, default=None,
                       help='Path to training data')
    parser.add_argument('--val_path', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--train_ann', type=str, default=None,
                       help='Path to training annotation file')
    parser.add_argument('--val_ann', type=str, default=None,
                       help='Path to validation annotation file')
    parser.add_argument('--backbone_weights', type=str, default=None,
                       help='Path to pretrained weights')
    parser.add_argument('--min_bbox_area', type=int, default=None,
                       help='Minimum bounding box area')
    
    return parser.parse_args()


def args_to_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    将命令行参数转换为配置字典
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        配置字典
    """
    config_dict = {}
    
    # 直接参数
    direct_params = [
        'lr', 'batch_size', 'max_epochs', 'num_workers',
        'min_bbox_area', 'backbone_weights', 'save_dir'
    ]
    for param in direct_params:
        value = getattr(args, param)
        if value is not None:
            config_dict[param] = value
    
    # 数据集参数
    dataset_params = {}
    if args.train_path is not None:
        dataset_params['train_path'] = args.train_path
    if args.val_path is not None:
        dataset_params['val_path'] = args.val_path
    if args.train_ann is not None:
        dataset_params['train_ann'] = args.train_ann
    if args.val_ann is not None:
        dataset_params['val_ann'] = args.val_ann
    
    if dataset_params:
        config_dict['dataset'] = dataset_params
    
    return config_dict
