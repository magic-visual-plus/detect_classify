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


def compute_iou(box1, box2):
    """
    计算两个bbox的IoU
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_confusion_matrix_binary(gts, preds, num_classes, score_thr=0.2, iou_thr = 0.5, eval_class_ids=None):
    """
    该函数通过遍历每一张图片的预测和标注，基于IoU和类别信息，统计三类情况的数量，最终返回混淆矩阵及相关指标
    混淆矩阵的结构如下：

                  | 预测为缺陷（正类） | 预测为正常或其他缺陷（负类） |
        ----------|-------------------|---------------------------|
        真实为缺陷 |        TP         |        FN                 |
        真实为正常 |        FP         |        TN                 |

    """
    conf_mat = np.zeros((num_classes, 2, 2), dtype=int)

    for i, (gt, pred) in enumerate(zip(gts, preds)):
        # 基于置信度和评估类别筛选出pred和gt
        if eval_class_ids is None:
            gt_cls_mask = np.ones(len(gt.cls), dtype=bool)
            pred_cls_mask = np.ones(len(pred.cls), dtype=bool)
        else:
            gt_cls_mask = np.array([_cls in eval_class_ids for _cls in gt.cls], dtype=bool)   # 筛选出评估类别
            pred_cls_mask = np.array([_cls in eval_class_ids for _cls in pred.cls], dtype=bool)
        pred_mask = (pred.scores > score_thr) & pred_cls_mask
        pred_bboxes = pred.bboxes[pred_mask]
        pred_cls = pred.cls[pred_mask]
        pred_scores = pred.scores[pred_mask]
        gt_bboxes = gt.bboxes[gt_cls_mask]
        gt_cls = gt.cls[gt_cls_mask]
        
        if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
            continue
        gt_matched = np.zeros(len(gt_bboxes), dtype=bool)
        for pred_i, (pred_bbox, pred_class_id) in enumerate(zip(pred_bboxes, pred_cls)):
            is_matched = False
            for gt_i, (gt_bbox, gt_class_id) in enumerate(zip(gt_bboxes, gt_cls)):
                if gt_class_id != pred_class_id:
                    continue
                if gt_matched[gt_i]:  
                    continue
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou > iou_thr:
                    conf_mat[gt_class_id, 0, 0] += 1
                    gt_matched[gt_i] = True
                    is_matched = True
                    break
            if not is_matched:
                conf_mat[pred_class_id, 1, 0] += 1
        for gt_mat, cls_id in zip(gt_matched, gt_cls):
            if not gt_mat:
                conf_mat[cls_id, 0, 1] += 1
    # 基于每个类别求平均（如果该类别的混淆矩阵全为0，则不计入平均）
    recalls = []
    precisions = []
    f1_scores = []
    for cls in range(conf_mat.shape[0]):
        # 判断该类别的混淆矩阵是否全为0
        if np.sum(conf_mat[cls]) == 0:
            continue
        tp = conf_mat[cls, 0, 0]
        fn = conf_mat[cls, 0, 1]
        fp = conf_mat[cls, 1, 0]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if (recall + precision) > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    recall_mean = np.mean(recalls) if len(recalls) > 0 else 0.0
    precision_mean = np.mean(precisions) if len(precisions) > 0 else 0.0
    f1_score_mean = np.mean(f1_scores) if len(f1_scores) > 0 else 0.0

    return {
        'conf_mat': conf_mat,
        'recall': recall_mean,
        'precision': precision_mean,
        'f1_score': f1_score_mean
    }


def compute_confusion_matrix_binary_ignore_class(gts, preds, score_thr=0.3, iou_thr=0.5, eval_class_ids=None):
    """
    计算二分类混淆矩阵（不考虑类别是否匹配，只要有IOU即可匹配）

    Args:
        gts: list，每个元素为gt记录，需有bboxes字段
        preds: list，每个元素为pred记录，需有bboxes和scores字段
        score_thr: 置信度阈值
        iou_thr: IOU阈值

    Returns:
        dict: 包含混淆矩阵、recall、precision、f1_score
    """
    tp = 0
    fp = 0
    fn = 0

    for gt, pred in zip(gts, preds):
        if eval_class_ids is None:
            gt_cls_mask = np.ones(len(gt.cls), dtype=bool)
            pred_cls_mask = np.ones(len(pred.cls), dtype=bool)
        else:
            gt_cls_mask = np.array([_cls in eval_class_ids for _cls in gt.cls], dtype=bool)   # 筛选出评估类别
            pred_cls_mask = np.array([_cls in eval_class_ids for _cls in pred.cls], dtype=bool)
        gt_bboxes = np.array(getattr(gt, 'bboxes', []))[gt_cls_mask]
        pred_bboxes = np.array(getattr(pred, 'bboxes', []))[pred_cls_mask]
        pred_scores = np.array(getattr(pred, 'scores', []))[pred_cls_mask]

        if len(pred_bboxes) == 0:
            fn += len(gt_bboxes)
            continue

        # 只保留大于score_thr的预测框
        keep = pred_scores > score_thr
        pred_bboxes = pred_bboxes[keep]
        if len(pred_bboxes) == 0:
            fn += len(gt_bboxes)
            continue

        matched_gt = np.zeros(len(gt_bboxes), dtype=bool)
        for pred_bbox in pred_bboxes:
            found_match = False
            for i, gt_bbox in enumerate(gt_bboxes):
                if matched_gt[i]:
                    continue
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > iou_thr:
                    tp += 1
                    matched_gt[i] = True
                    found_match = True
                    break
            if not found_match:
                fp += 1
        fn += np.sum(~matched_gt)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score
    }


class DetectionClassificationEvaluator:
    """two stage模型评估器"""
    
    def __init__(self, 
                 detection_model_path: str,
                 classification_model_path: str,
                 backbone_name: str,
                 device: str = "cuda:0",
                 repo_dir: str = "/root/dinov3",
                 min_area: int = 900,
                 classification_confidence_threshold: float = 0.3,
                 classification_pad_color: Tuple[int, int, int] = (114, 114, 114),
                 classification_class_names: Optional[List[str]] = None):
        """
        初始化评估器
        
        Args:
            detection_model_path: 检测模型路径
            classification_model_path: 分类模型路径
            backbone_name: 分类模型backbone名称
            device: 设备
            repo_dir: DINOv3仓库目录
            min_area: 分类过滤最小面积
            classification_confidence_threshold: 分类过滤检测置信度阈值
            classification_pad_color: 分类模型填充颜色
            classification_class_names: 分类模型类别名称
        """
        self.device = device
        self.repo_dir = repo_dir
        self.min_area = min_area
        self.classification_confidence_threshold = classification_confidence_threshold
        self.classification_pad_color = classification_pad_color
        self.classification_class_names = classification_class_names
        
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
        if self.classification_class_names is not None:
            print(f"分类模型类别名称: {self.classification_class_names}")
    
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
        if self.classification_class_names is not None:
            cls_config['class_names'] = self.classification_class_names
        
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
            (res is not None) and (res['all_scores'][filter_class_id] > score_threshold)
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
                            classification_score_threshold: float = 0.2,
                            min_area: Optional[int] = None,
                            classification_confidence_threshold: Optional[float] = None) -> Tuple[PredictionResult, PredictionResult, int]:
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
            # 根据id2names找到classification_class_names在检测模型中的类别id
            classification_class_ids = []
            for class_name in self.classification_class_names:
                found = False
                for class_id, name in self.detection_model.id2names.items():
                    if name == class_name:
                        classification_class_ids.append(class_id)
                        found = True
                        break
                if not found:
                    print(f"Warning: '{class_name}' not found in detection model id2names")
            min_area_val = min_area if min_area is not None else self.min_area
            conf_thr = classification_confidence_threshold if classification_confidence_threshold is not None else self.classification_confidence_threshold
            areas = (detection_result.bboxes[:, 2] - detection_result.bboxes[:, 0]) * (detection_result.bboxes[:, 3] - detection_result.bboxes[:, 1])
            conf_mask = (detection_result.scores > conf_thr) & (areas > min_area_val) & (np.isin(detection_result.cls, np.array(classification_class_ids)))
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
                        final_score_threshold: float = 0.3,
                        min_area: Optional[int] = None,
                        classification_confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
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
            min_area: 分类过滤最小面积
            classification_confidence_threshold: 分类过滤检测置信度阈值
            
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
                classification_score_threshold=classification_score_threshold,
                min_area=min_area,
                # classification_confidence_threshold=classification_confidence_threshold
                classification_confidence_threshold=0.2
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
        
        classification_confidence_threshold_result = compute_confusion_matrix_binary_ignore_class(gts, preds, score_thr=classification_confidence_threshold, iou_thr=0.5, eval_class_ids=eval_class_ids)
        print(f"confidence_threshold: {classification_confidence_threshold}")
        print(classification_confidence_threshold_result)
        print('-'*80)

        # scores_thr = np.arange(0.2, 0.5, 0.01)
        # f1s = np.zeros(len(scores_thr))
        # resluts = []
        # f1s_ignore_class = np.zeros(len(scores_thr))
        # ignore_class_results = []
        # for i, score_thr in tqdm(enumerate(scores_thr), desc="评估中..."):
            # res = compute_confusion_matrix_binary(gts, preds, num_classes=len(self.class_names), score_thr=score_thr, iou_thr=0.5, eval_class_ids=eval_class_ids)
            # f1 = res['f1_score']
            # resluts.append(res)
            # f1s[i] = f1
            # ignore_class_res = compute_confusion_matrix_binary_ignore_class(gts, preds, score_thr=score_thr, iou_thr=0.5, eval_class_ids=eval_class_ids)
            # ignore_class_results.append(ignore_class_res)
            # f1s_ignore_class[i] = ignore_class_res['f1_score']
        # max_id = np.argmax(f1s)
        # print("考虑类别匹配的评估结果:")
        # print(f"confidence: {scores_thr[max_id]}, recall: {resluts[max_id]['recall']}, precision: {resluts[max_id]['precision']}, f1_score: {resluts[max_id]['f1_score']}")
        # print("不考虑类别匹配的评估结果:")
        # max_id_ignore_class = np.argmax(f1s_ignore_class)
        # print(f"confidence: {scores_thr[max_id_ignore_class]}, {ignore_class_results[max_id_ignore_class]}")
        # print(ignore_class_results)
       
        # 计算设定评估项的评估结果
        class_specific_results = evaluate.eval_detection_result_by_class_id(gts, preds, eval_class_ids)
        conf = class_specific_results['confidence']

        return {
            'class_specific_results': class_specific_results,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'classification_confidence_threshold_result': {'confidence_threshold': classification_confidence_threshold, 'result': classification_confidence_threshold_result}
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
    parser.add_argument('--fixed_detection_confidence', action='store_true', help='检测模型是否始终使用固定置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IOU阈值')
    parser.add_argument('--max_size', type=int, default=1536, help='最大图像尺寸')
    parser.add_argument('--classification_scale_factor', type=float, default=1.0, help='分类图片缩放因子')
    parser.add_argument('--classification_score_threshold', type=float, default=0.6, help='分类分数阈值')
    parser.add_argument('--final_score_threshold', type=float, default=0.3, help='最终分数阈值')
    parser.add_argument('--filter_class_id', type=int, help='要过滤的类别ID')
    parser.add_argument('--no_classification_filter', action='store_true', help='不使用分类过滤')
    parser.add_argument('--eval_class_names', nargs='*', help='要评估的类别名称')
    parser.add_argument('--min_area', type=int, default=0, help='最小面积')
    parser.add_argument('--classification_confidence_threshold', type=float, default=0.25, help='分类模型过滤检测置信度阈值')
    parser.add_argument('--classification_pad_color', type=str, default="114,114,114", help='分类模型填充颜色, 逗号分隔如114,114,114')
    parser.add_argument('--classification_class_names', nargs='*', help='分类模型类别名称')
    
    args = parser.parse_args()
    if args.eval_class_names is None:
        args.eval_class_names = ['划伤', '压痕', '异物外漏', '折痕', '抛线', "吊紧", "烫伤", 
        '拼接间隙', '破损', '碰伤', '红标签', '线头', '脏污', '褶皱(T型)', '褶皱（重度）', '重跳针']
    if args.classification_class_names is None:
        args.classification_class_names = args.eval_class_names
    # 解析分类模型填充颜色
    if isinstance(args.classification_pad_color, str):
        try:
            pad_color = tuple(int(x) for x in args.classification_pad_color.split(','))
            if len(pad_color) != 3:
                raise ValueError
        except Exception:
            print("分类模型填充颜色参数格式错误，应为如114,114,114")
            pad_color = (114, 114, 114)
    else:
        pad_color = args.classification_pad_color
    print("------"* 10)
    print(args)
    print("------" * 10)

    # 创建评估器
    evaluator = DetectionClassificationEvaluator(
        detection_model_path=args.detection_model,
        classification_model_path=args.classification_model,
        backbone_name=args.backbone_name,
        device=args.device,
        repo_dir=args.repo_dir,
        min_area=args.min_area,
        classification_confidence_threshold=args.classification_confidence_threshold,
        classification_pad_color=pad_color,
        classification_class_names=args.classification_class_names
    )
    # 执行评估
    # results = evaluator.evaluate_dataset(
    #     input_path=args.input_path,
    #     confidence_threshold=args.confidence_threshold,
    #     max_size=args.max_size,
    #     classification_scale_factor=args.classification_scale_factor,
    #     classification_pad_color=pad_color,
    #     filter_by_classification=not args.no_classification_filter,
    #     filter_class_id=args.filter_class_id,
    #     classification_score_threshold=args.classification_score_threshold,
    #     eval_class_names=args.eval_class_names,
    #     final_score_threshold=args.final_score_threshold,
    #     min_area=args.min_area,
    #     classification_confidence_threshold=args.classification_confidence_threshold
    # )
    # evaluator.print_results(results)
    # exit()
    test_results = {}
    for classification_score_threshold in np.arange(0.00, 1.00, 0.05): #  0.7, 0.8, 0.9]:
        results = evaluator.evaluate_dataset(
            input_path=args.input_path,
            confidence_threshold=args.confidence_threshold,
            max_size=args.max_size,
            classification_scale_factor=args.classification_scale_factor,
            classification_pad_color=pad_color,
            filter_by_classification=not args.no_classification_filter,
            filter_class_id=args.filter_class_id,
            # classification_score_threshold=args.classification_score_threshold,
            classification_score_threshold=classification_score_threshold,
            eval_class_names=args.eval_class_names,
            final_score_threshold=args.final_score_threshold,
            min_area=args.min_area,
            classification_confidence_threshold=args.classification_confidence_threshold
        )
        test_results[classification_score_threshold] = results
    return evaluator, test_results

if __name__ == '__main__':
    evaluator, results = main()
    for classification_score_threshold, result in results.items():
        print(f"classification_score: {classification_score_threshold}")
        print(result['classification_confidence_threshold_result'])
    
    for classification_score_threshold, result in results.items():
        print(f"classification_score: {classification_score_threshold}")
        print(result['class_specific_results'])
        # evaluator.print_results(result)
