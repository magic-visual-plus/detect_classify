"""
检测和分类模型评估工具
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hq_det.models.dino import hq_dino
from hq_det.dataset import CocoDetection
from hq_det.common import PredictionResult
from hq_det import augment, evaluate

from trainer import get_model
import utils
import utils.transforms


class DetectionClassificationEvaluator:
    """two stage模型评估器"""
    
    def __init__(self, 
                 detection_model_path: str,
                 classification_model_path: str,
                 backbone_name: str,
                 device: str = "cuda:0",
                 repo_dir: str = "/root/dinov3"):
        """
        初始化评估器
        
        Args:
            detection_model_path: 检测模型路径
            classification_model_path: 分类模型路径
            backbone_name: 分类模型backbone名称
            device: 设备
            repo_dir: DINOv3仓库目录
        """
        self.device = device
        self.repo_dir = repo_dir
        
        # 加载检测模型
        self.detection_model = self._load_detection_model(detection_model_path)
        
        # 加载分类模型
        self.classification_model = self._load_classification_model(
            classification_model_path, backbone_name
        )
        
        # 获取类别信息
        self.class_names = self.detection_model.get_class_names()
        self.num_classes = self.classification_model.num_classes if hasattr(self.classification_model, 'num_classes') else len(self.class_names)
        
        print(f"检测模型类别数: {len(self.class_names)}")
        print(f"分类模型类别数: {self.num_classes}")
        print(f"检测模型类别名称: {self.class_names}")
    
    def _load_detection_model(self, model_path: str):
        """加载检测模型"""
        print(f"正在加载检测模型: {model_path}")
        model = hq_dino.HQDINO(model=model_path)
        model.eval()
        model.to(self.device)
        return model
    
    def _load_classification_model(self, model_path: str, backbone_name: str):
        """加载分类模型"""
        print(f"正在加载分类模型: {model_path}")
        
        # 从checkpoint推断类别数
        checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # 获取最后一个参数的shape来推断类别数
        last_param_name = list(state_dict.keys())[-1]
        last_param = state_dict[last_param_name]
        num_classes = max(last_param.shape[0], 2)
        print(num_classes)
        
        # 创建分类模型配置
        cls_config = {
            'check_point_path': model_path,
            "backbone_name": backbone_name,
            'REPO_DIR': self.repo_dir,
            'num_classes': num_classes
        }
        
        # 创建并加载模型
        model = get_model(cls_config)
        model.eval()
        model.to(self.device)
        
        return model
    
    def _prepare_transforms(self):
        """准备数据变换"""
        transforms = [augment.ToNumpy()]
        return augment.Compose(transforms)
    
    def _load_dataset(self, input_path: str):
        """加载数据集"""
        transforms = self._prepare_transforms()
        dataset = CocoDetection(
            input_path,
            os.path.join(input_path, '_annotations.coco.json'),
            transforms=transforms
        )
        return dataset
    
    def _filter_predictions_by_classification(self, 
                                            detection_result: PredictionResult,
                                            classification_results: List[Dict],
                                            filter_class_id: Optional[int] = None,
                                            score_threshold: float = 0.2) -> Tuple[PredictionResult, int]:
        """
        根据分类结果过滤检测预测，并返回被过滤的数量
        
        Args:
            detection_result: 检测结果
            classification_results: 分类结果
            filter_class_id: 要过滤的类别ID（如果为None，则过滤最后一个类别）
            score_threshold: 分数阈值
            
        Returns:
            过滤后的检测结果, 被过滤的数量
        """
        if not classification_results:
            return detection_result, 0
        
        # 确定要过滤的类别ID
        if filter_class_id is None:
            filter_class_id = self.num_classes - 1
        
        # 创建过滤掩码
        cls_mask = [
            (res is not None) and (res.get('pred_class') == filter_class_id) and (res.get('score', 0) > score_threshold)
            for res in classification_results
        ]
        
        # 应用过滤
        cls_mask_np = np.array(cls_mask)
        keep_indices = np.where(~cls_mask_np)[0]
        num_filtered = len(detection_result.bboxes) - len(keep_indices)
        
        if len(keep_indices) < len(detection_result.bboxes):
            detection_result.bboxes = detection_result.bboxes[keep_indices]
            detection_result.scores = detection_result.scores[keep_indices]
            detection_result.cls = detection_result.cls[keep_indices]
        
        return detection_result, num_filtered
    
    def _process_single_image(self, 
                            img: np.ndarray, 
                            bboxes: np.ndarray, 
                            cls: np.ndarray, 
                            image_id: Any,
                            confidence_threshold: float = 0.0,
                            max_size: int = 1536,
                            classification_scale_factor: float = 1.0,
                            classification_pad_color: Tuple[int, int, int] = (114, 114, 114),
                            filter_by_classification: bool = True,
                            filter_class_id: Optional[int] = None,
                            classification_score_threshold: float = 0.2) -> Tuple[PredictionResult, PredictionResult, int]:
        """
        处理单张图像
        
        Returns:
            (ground_truth, prediction, num_filtered)
        """
        # 创建ground truth记录
        gt_record = PredictionResult()
        gt_record.bboxes = bboxes
        gt_record.cls = cls
        gt_record.image_id = image_id
        gt_record.scores = np.ones(len(bboxes), dtype=np.float32)
        
        # 检测预测
        start_time = time.time()
        detection_results = self.detection_model.predict(
            [img], bgr=True, confidence=confidence_threshold, max_size=max_size
        )
        detection_time = time.time() - start_time
        
        detection_result = detection_results[0]
        
        num_filtered = 0
        # 分类预测和过滤
        if filter_by_classification and len(detection_result.bboxes) > 0:
            # 只对检测置信度大于阈值且面积大于阈值的框做分类预测
            min_area = 900
            areas = (detection_result.bboxes[:, 2] - detection_result.bboxes[:, 0]) * (detection_result.bboxes[:, 3] - detection_result.bboxes[:, 1])
            conf_mask = (detection_result.scores > 0.3) & (areas > min_area)
            if np.any(conf_mask):
                # 只取高置信度的框
                filtered_bboxes = detection_result.bboxes[conf_mask]
                # 准备图像用于分类
                if isinstance(img, np.ndarray):
                    img_pil = Image.fromarray(img)
                else:
                    img_pil = img

                # 分类预测
                classification_results = self.classification_model.predict(
                    img_pil,
                    filtered_bboxes,
                    scale_factor=classification_scale_factor,
                    pad_mode="constant",
                    pad_color=classification_pad_color,
                    transformer=utils.transforms.get_default_transform(size=(224, 224))
                )
                print(f"classification_results: {classification_results}")

                # 只对高置信度框做分类过滤，低置信度框直接保留
                # 构造一个和原检测框数量一致的分类结果列表，低置信度的填None
                full_classification_results = [None] * len(detection_result.bboxes)
                idx_high_conf = np.where(conf_mask)[0]
                for i, idx in enumerate(idx_high_conf):
                    full_classification_results[idx] = classification_results[i]

                # 根据分类结果过滤检测结果
                detection_result, num_filtered = self._filter_predictions_by_classification(
                    detection_result,
                    full_classification_results,
                    filter_class_id=filter_class_id,
                    score_threshold=classification_score_threshold
                )
            # 如果没有高置信度框，则不做分类过滤，直接返回
        detection_result.image_id = image_id
        
        return gt_record, detection_result, num_filtered
    
    def evaluate_dataset(self, 
                        input_path: str,
                        confidence_threshold: float = 0.0,
                        max_size: int = 1536,
                        classification_scale_factor: float = 1.0,
                        classification_pad_color: Tuple[int, int, int] = (114, 114, 114),
                        filter_by_classification: bool = True,
                        filter_class_id: Optional[int] = None,
                        classification_score_threshold: float = 0.2,
                        eval_class_names: Optional[List[str]] = None,
                        final_score_threshold: float = 0.3) -> Dict[str, Any]:
        """
        评估整个数据集
        
        Args:
            input_path: 输入数据路径
            confidence_threshold: 检测置信度阈值
            max_size: 最大图像尺寸
            classification_scale_factor: 分类时的缩放因子
            classification_pad_color: 分类时的填充颜色
            filter_by_classification: 是否使用分类结果过滤
            filter_class_id: 要过滤的类别ID
            classification_score_threshold: 分类分数阈值
            eval_class_names: 要评估的类别名称列表
            final_score_threshold: 最终分数阈值
            
        Returns:
            评估结果字典
        """
        print(f"正在加载数据集: {input_path}")
        dataset = self._load_dataset(input_path)
        
        print(f"数据集大小: {len(dataset)}")
        print("开始评估...")
        
        preds = []
        gts = []

        total_filtered = 0  # 累计被过滤的框数量

        def tqdm_with_filtered(iterable, **kwargs):
            pbar = tqdm(iterable, **kwargs)
            pbar.set_postfix({'cur_filtered': 0, 'total_filtered': 0})
            return pbar

        pbar = tqdm_with_filtered(range(len(dataset)), desc="评估进度")
        for idx in pbar:
            data = dataset[idx]
            
            gt_record, pred_record, num_filtered = self._process_single_image(
                img=data['img'],
                bboxes=data['bboxes'],
                cls=data['cls'],
                image_id=data['image_id'],
                confidence_threshold=confidence_threshold,
                max_size=max_size,
                classification_scale_factor=classification_scale_factor,
                classification_pad_color=classification_pad_color,
                filter_by_classification=filter_by_classification,
                filter_class_id=filter_class_id,
                classification_score_threshold=classification_score_threshold
            )
            
            gts.append(gt_record)
            preds.append(pred_record)
            total_filtered += num_filtered
            pbar.set_postfix({'cur_filtered': num_filtered, 'total_filtered': total_filtered})
        
        # 根据评估项（eval_class_names）进行最终的评估
        print("\n计算评估指标...")

        # 获取需要评估的类别ID
        eval_class_ids = []
        if eval_class_names:
            eval_class_ids = [self.class_names.index(c) for c in eval_class_names if c in self.class_names]

        # 计算设定评估项的评估结果
        class_specific_results = evaluate.eval_detection_result_by_class_id(gts, preds, eval_class_ids)

        return {
            'class_specific_results': class_specific_results,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }

    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print("\n" + "="*80)
        print("检测和分类模型评估结果")
        print("="*80)

        print("\n特定类别评估结果:")
        print("-" * 50)
        print(results['class_specific_results'])

        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检测和分类模型评估工具')
    parser.add_argument('detection_model', help='检测模型路径')
    parser.add_argument('input_path', help='输入数据路径')
    parser.add_argument('classification_model', help='分类模型路径')
    parser.add_argument('backbone_name', help='分类模型backbone名称')
    parser.add_argument('--device', default='cuda:0', help='设备 (默认: cuda:0)')
    parser.add_argument('--repo_dir', default='/root/dinov3', help='DINOv3仓库目录')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, help='检测置信度阈值')
    parser.add_argument('--max_size', type=int, default=1536, help='最大图像尺寸')
    parser.add_argument('--classification_scale_factor', type=float, default=1.0, help='分类图片缩放因子')
    parser.add_argument('--classification_score_threshold', type=float, default=0.9, help='分类分数阈值')
    parser.add_argument('--final_score_threshold', type=float, default=0.3, help='最终分数阈值')
    parser.add_argument('--filter_class_id', type=int, help='要过滤的类别ID')
    parser.add_argument('--no_classification_filter', action='store_true', help='不使用分类过滤')
    parser.add_argument('--eval_class_names', nargs='*', help='要评估的类别名称')
    
    args = parser.parse_args()
    if args.eval_class_names is None:
        args.eval_class_names =  [
            '划伤', '压痕', '异物外漏', '折痕', '抛线', 
            '拼接间隙', '破损', '碰伤', '红标签', '线头', 
            '脏污', '褶皱(T型)', '褶皱（重度）', '重跳针'
        ]
    
    # 创建评估器
    evaluator = DetectionClassificationEvaluator(
        detection_model_path=args.detection_model,
        classification_model_path=args.classification_model,
        backbone_name=args.backbone_name,
        device=args.device,
        repo_dir=args.repo_dir
    )
    
    # 执行评估
    results = evaluator.evaluate_dataset(
        input_path=args.input_path,
        confidence_threshold=args.confidence_threshold,
        max_size=args.max_size,
        classification_scale_factor=args.classification_scale_factor,
        filter_by_classification=not args.no_classification_filter,
        filter_class_id=args.filter_class_id,
        classification_score_threshold=args.classification_score_threshold,
        eval_class_names=args.eval_class_names,
        final_score_threshold=args.final_score_threshold
    )
    
    # 打印结果
    evaluator.print_results(results)


if __name__ == '__main__':
    main()