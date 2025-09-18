import os
import json
import copy
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from torchvision import transforms
from utils.hq_dino_feature_extractor import DinoFeatureExtractor


__all__ = [
    "COCOClassificationDataset",    # COCO分类任务数据集
    "crop_adaptive_square",          # 自适应正方形裁剪
    "load_coco_file",               # 加载coco文件
    "save_coco_file",               # 保存cocow文件
    "filter_and_remap_coco_categories",  # 筛选并重映射COCO类别
    "correct_predicted_categories",      # 修正预测框类别
    "merge_coco_categories"              # 合并COCO类别
    "filter_coco_by_source_category"      # 筛选COCO数据
]


class COCOClassificationDataset(Dataset):
    """
    COCO数据集的分类任务Dataset类
    将每个标注框裁剪出来作为一个独立的分类样本
    """

    def __init__(self, 
                 root_dir, 
                 ann_file, 
                 transform=None,
                 min_bbox_area=0,
                 crop_scale_factor=1.0,
                 pad_mode="constant",
                 pad_color=(114, 114, 114),
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
        self.crop_scale_factor = crop_scale_factor
        self.pad_mode = pad_mode
        self.pad_color = pad_color
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

        # 获取所有有效的标注（annotation），只保留属于目标类别的标注
        self.annotations = []
        for cat_id in self.cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=[cat_id])
            anns = self.coco.loadAnns(ann_ids)
            # 过滤掉太小的边界框
            for ann in anns:
                # 只保留属于目标类别的标注
                if ann['area'] >= self.min_bbox_area and ann['category_id'] in self.cat_id_to_label:
                    self.annotations.append(ann)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        返回裁剪后的图片和对应的类别标签
        """
        # 获取标注信息
        ann = self.annotations[idx]

        category_id = ann['category_id']
        if category_id not in self.cat_id_to_label:
            raise ValueError(f"标注类别ID {category_id} 不在目标类别列表中，请检查数据集和类别过滤设置。")

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

        cropped_image = crop_adaptive_square(image, x, y, w, h, scale_factor=self.crop_scale_factor, pad_mode=self.pad_mode, pad_color=self.pad_color)

        label = self.cat_id_to_label[category_id]
        if ann.get('feat_path', None) is not None:
            feat_path = ann['feat_path']
            feats = DinoFeatureExtractor.load_feats_from_npz(feat_path)
            pooled_feats = []
            for feat in feats:
                feat_width, feat_height = feat.shape[1], feat.shape[2]
                feat_x1 = int(x * feat_width / img_width)
                feat_y1 = int(y * feat_height / img_height)
                feat_x2 = min(int(1 + x2 * feat_width / img_width), feat_width)  
                feat_y2 = min(int(1 + y2 * feat_height / img_height), feat_height)  
                crop_feat = feat[:, feat_y1:feat_y2, feat_x1:feat_x2]
                if not isinstance(crop_feat, torch.Tensor):
                    crop_feat = torch.from_numpy(crop_feat)
                pooled_feat = crop_feat.amax(dim=(1,2))
                pooled_feats.append(pooled_feat) 
        pooled_feats = torch.stack(pooled_feats, dim=0)
        if self.transform:
            cropped_image = self.transform(cropped_image)
        info = {
            'image_id': ann['image_id'],
            'annotation_id': ann['id'],
            'category_name': self.label_to_cat_name[label],
            'bbox': bbox,
            'original_size': (img_width, img_height),
            'image_name': img_info['file_name'],
            'annotation_source': ann.get('source', None),
            'detection_feature': pooled_feats
        }

        return cropped_image, label, info

    def export_to_folder(self, export_dir, img_format="jpg", exist_ok=False, scale_factor=1.0, pad_mode="constant", pad_color=(114, 114, 114)):
        """
        Export the dataset as a folder-structured classification dataset,
        with each class in a subfolder and images cropped to the target object.
        The exported image filename will include the annotation source if available.
        Args:
            export_dir (str): Root directory to export to.
            img_format (str): Image format for export (e.g., 'jpg', 'png').
            exist_ok (bool): Whether to allow overwriting if the directory exists.
            scale_factor (float): Scale factor for cropping.
        """
        import shutil
        from tqdm import tqdm

        # PIL image format mapping
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "bmp": "BMP",
            "tiff": "TIFF",
            "webp": "WEBP"
        }
        img_format_lower = img_format.lower()
        pil_format = format_map.get(img_format_lower, None)
        if pil_format is None:
            raise ValueError(f"Unsupported image format: {img_format}. Please use jpg, png, bmp, tiff, webp, etc.")

        if os.path.exists(export_dir):
            if not exist_ok:
                raise FileExistsError(f"Export directory {export_dir} already exists. Set exist_ok=True to overwrite.")
            else:
                print(f"Warning: Export directory {export_dir} already exists. Contents may be overwritten.")
        else:
            os.makedirs(export_dir, exist_ok=True)

        # Create subdirectories for each class
        for label in range(len(self.cat_ids)):
            class_name = self.label_to_cat_name[label]
            class_dir = os.path.join(export_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        for idx in tqdm(range(len(self.annotations)), desc="Exporting classification images"):
            ann = self.annotations[idx]
            category_id = ann['category_id']
            label = self.cat_id_to_label[category_id]
            class_name = self.label_to_cat_name[label]

            img_info = self.coco.loadImgs(ann['image_id'])[0]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            image = Image.open(img_path).convert('RGB')

            bbox = ann['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cropped_image = crop_adaptive_square(image, x, y, w, h,  scale_factor=scale_factor, pad_mode=pad_mode, pad_color=pad_color)

            # Compose filename: imageid_annid_source.suffix (source is optional, suffix is always lower case)
            source = ann.get('source', None)
            if source is not None and str(source).strip() != "":
                out_name = f"{ann['image_id']}_{idx}_{source}.{img_format_lower}"
            else:
                out_name = f"{ann['image_id']}_{idx}.{img_format_lower}"
            out_path = os.path.join(export_dir, class_name, out_name)
            cropped_image.save(out_path, format=pil_format)

        print(f"Export complete. Images saved to {export_dir}")

    def get_num_classes(self):
        """返回类别数量"""
        return len(self.cat_ids)

    def get_class_names(self):
        """返回类别名称列表"""
        return [self.label_to_cat_name[i] for i in range(len(self.cat_ids))]

def crop_adaptive_square(image, x, y, w, h, scale_factor=1.5, pad_mode="constant", pad_color=(114, 114, 114)):  
    """  
    自适应正方形裁剪，根据原区域尺寸动态确定正方形大小。  
    
    参数：  
        image: PIL.Image 对象  
        x, y, w, h: 原始区域的左上角坐标和宽高  
        scale_factor: 短边放大倍数 n  
        pad_mode: 越界填充方式 ("constant", "edge", None)  
        pad_color: 常数填充的颜色 (R, G, B)  
    
    返回：  
        PIL.Image，裁剪后的正方形图像  
    """  
    img_w, img_h = image.size  
    
    # 计算原区域的短边和长边  
    short_side = min(w, h)  
    long_side = max(w, h)  
    
    # 确定正方形边长：max(长边, 短边 * scale_factor)  
    square_size = max(long_side, int(short_side * scale_factor))  
    
    # 原区域中心点  
    cx = x + w / 2.0  
    cy = y + h / 2.0  
    
    # 以中心为基准的正方形裁剪框  
    half_size = square_size / 2.0  
    left = int(round(cx - half_size))  
    top = int(round(cy - half_size))  
    right = left + square_size  
    bottom = top + square_size  
    
    # 检查越界  
    overflow_left = max(0, -left)  
    overflow_top = max(0, -top)  
    overflow_right = max(0, right - img_w)  
    overflow_bottom = max(0, bottom - img_h)  
    
    if any([overflow_left, overflow_top, overflow_right, overflow_bottom]):  
        if pad_mode is None:  
            # 仅在有效范围内裁剪  
            left_clip = max(0, left)  
            top_clip = max(0, top)  
            right_clip = min(img_w, right)  
            bottom_clip = min(img_h, bottom)  
            return image.crop((left_clip, top_clip, right_clip, bottom_clip))  
        
        elif pad_mode == "constant":  
            if isinstance(pad_color, (list, tuple)):
                pad_color = tuple(int(c) for c in pad_color)
            square_size = int(square_size)
            canvas = Image.new(image.mode, (square_size, square_size), pad_color)  
            
            # 计算有效区域  
            src_left = max(0, left)  
            src_top = max(0, top)  
            src_right = min(img_w, right)  
            src_bottom = min(img_h, bottom)  
            
            if src_right > src_left and src_bottom > src_top:  
                patch = image.crop((src_left, src_top, src_right, src_bottom))  
                canvas.paste(patch, (overflow_left, overflow_top))  
            
            return canvas  
        
        elif pad_mode == "edge":   
            return _crop_with_edge_padding(image, left, top, square_size, square_size)  
    
    else:  
        # 不越界，直接裁剪  
        return image.crop((left, top, right, bottom))  


def _crop_with_edge_padding(image, left, top, target_w, target_h):  
    """边缘延展填充"""  
    img_w, img_h = image.size  
    img_array = np.array(image)  
    
    # 创建输出数组  
    if len(img_array.shape) == 3:  
        output = np.zeros((target_h, target_w, img_array.shape[2]), dtype=img_array.dtype)  
    else:  
        output = np.zeros((target_h, target_w), dtype=img_array.dtype)  
    
    # 计算有效区域  
    src_left = max(0, left)  
    src_top = max(0, top)  
    src_right = min(img_w, left + target_w)  
    src_bottom = min(img_h, top + target_h)  
    
    dst_left = src_left - left  
    dst_top = src_top - top  
    dst_right = dst_left + (src_right - src_left)  
    dst_bottom = dst_top + (src_bottom - src_top)  
    
    # 复制有效区域  
    if src_right > src_left and src_bottom > src_top:  
        output[dst_top:dst_bottom, dst_left:dst_right] = img_array[src_top:src_bottom, src_left:src_right]  
        
        if dst_top > 0:  
            output[:dst_top, dst_left:dst_right] = output[dst_top:dst_top+1, dst_left:dst_right]  
        if dst_bottom < target_h:  
            output[dst_bottom:, dst_left:dst_right] = output[dst_bottom-1:dst_bottom, dst_left:dst_right]  
        if dst_left > 0:  
            output[:, :dst_left] = output[:, dst_left:dst_left+1]  
        if dst_right < target_w:  
            output[:, dst_right:] = output[:, dst_right-1:dst_right]  
    
    return Image.fromarray(output)  

def filter_coco_by_source_category(coco_data, source_values):
    """
    只筛选出所有source为指定值（如"predict"）的标注和相关图片。

    Args:
        coco_data: COCO格式的字典数据
        source_values: 可以是字符串或字符串列表

    Returns:
        new_coco_data: 只包含指定source的标注和相关图片的COCO数据
    """
    # 兼容单个字符串
    if isinstance(source_values, str):
        source_values = [source_values]

    # 只保留source为指定值的标注
    filtered_annotations = [
        ann for ann in coco_data['annotations']
        if ann.get('source', None) in source_values
    ]

    # 只保留有标注的图片
    used_image_ids = set(ann['image_id'] for ann in filtered_annotations)
    filtered_images = [img for img in coco_data['images'] if img['id'] in used_image_ids]

    # 类别全部保留
    filtered_categories = copy.deepcopy(coco_data['categories'])

    # 构建新的COCO数据
    new_coco_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }
    # 保留其他字段（如info、licenses等）
    for k in coco_data:
        if k not in new_coco_data:
            new_coco_data[k] = copy.deepcopy(coco_data[k])

    return new_coco_data



def load_coco_file(coco_file) -> dict:
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    return coco_data

def save_coco_file(coco_data: dict, coco_file: str):
    with open(coco_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

def filter_and_remap_coco_categories(coco_data, target_category_names):
    """
    根据指定类别名称筛选COCO数据，并将类别id重新排序
    Args:
        coco_data: COCO格式的字典数据
        target_category_names: 需要保留的类别名称列表
    Returns:
        coco_data
    """
    import copy

    # 找到目标类别的原始id和名称
    name2cat = {cat['name']: cat for cat in coco_data['categories']}
    filtered_cats = [name2cat[name] for name in target_category_names if name in name2cat]
    if not filtered_cats:
        raise ValueError("未找到任何目标类别，请检查类别名称是否正确。")

    # 重新分配类别id
    new_categories = []
    oldid2newid = {}
    for new_id, cat in enumerate(filtered_cats):
        new_cat = {
            "id": new_id,
            "name": cat["name"],
            "supercategory": cat.get("supercategory", "none")
        }
        new_categories.append(new_cat)
        oldid2newid[cat["id"]] = new_id

    # 筛选并重映射标注
    new_annotations = []
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in oldid2newid:
            new_ann = copy.deepcopy(ann)
            new_ann['category_id'] = oldid2newid[old_cat_id]
            new_annotations.append(new_ann)

    # 保留所有图片
    used_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in coco_data['images'] if img['id'] in used_image_ids]

    # 构建新的coco_data
    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }
    for k in coco_data:
        if k not in new_coco:
            new_coco[k] = copy.deepcopy(coco_data[k])

    return new_coco


def correct_predicted_categories(coco_data, iou_threshold=0.5):
    """
    修正预测框的类别：
    - 如果预测框与真实框的IOU低于阈值，则类别标记为背景
    - 如果IOU高于阈值但类别不一致，则纠正为真实框的类别
    同时自动添加背景类别，id为最大类别id+1，并写入coco_data['categories']

    Args:
        coco_data: COCO格式数据，包含images, annotations, categories
        iou_threshold: IOU阈值
    Returns:
        修正后的coco_data
    """
    if not isinstance(iou_threshold, float):
        return coco_data
    # 获取当前最大类别id
    if len(coco_data['categories']) == 0:
        max_cat_id = 0
    else:
        max_cat_id = max(cat['id'] for cat in coco_data['categories'])
    background_id = max_cat_id + 1

    has_background = any(cat['id'] == background_id or cat['name'] == "背景" for cat in coco_data['categories'])
    if not has_background:
        coco_data['categories'].append({
            "id": background_id,
            "name": "背景",
            "supercategory": "none"
        })
    
    def compute_iou(box1, box2):
        # box: [x, y, w, h]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    # 按image_id分组
    gt_anns = {}
    pred_anns = {}
    for ann in coco_data["annotations"]:
        if ann.get("source", "") == "predict":
            pred_anns.setdefault(ann["image_id"], []).append(ann)
        else:
            gt_anns.setdefault(ann["image_id"], []).append(ann)

    for image_id, preds in pred_anns.items():
        gts = gt_anns.get(image_id, [])
        for pred in preds:
            max_iou = 0
            matched_gt = None
            for gt in gts:
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    matched_gt = gt
            if max_iou < iou_threshold:
                pred["category_id"] = background_id
            elif matched_gt and pred["category_id"] != matched_gt["category_id"]:
                pred["category_id"] = matched_gt["category_id"]
    return coco_data

def merge_coco_categories(coco_data, merge_dict = None):
    """
    合并COCO标签类别

    参数:
        coco_data: COCO格式的字典数据
        merge_dict: dict，key为合并后的新类别名，value为要合并的原类别名列表

    Return:
        合并后的coco_data
    """
    if not isinstance(merge_dict, dict):
        return coco_data
    print(merge_dict)
    name2id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    new_categories = []
    new_name2id = {}
    for idx, (new_name, old_names) in enumerate(merge_dict.items()):
        new_categories.append({
            "id": idx,
            "name": new_name,
            "supercategory": "none"
        })
        new_name2id[new_name] = idx
    oldid2newid = {}
    for new_name, old_names in merge_dict.items():
        for old_name in old_names:
            if old_name not in name2id:
                print(f"warning: not find {old_name}")
                continue
            oldid2newid[name2id[old_name]] = new_name2id[new_name]
    for ann in coco_data['annotations']:
        old_id = ann['category_id']
        if old_id in oldid2newid:
            ann['category_id'] = oldid2newid[old_id]
        else:
            ann['category_id'] = -1 
    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['category_id'] != -1]
    coco_data['categories'] = new_categories
    return coco_data

if __name__ == "__main__":
    train_root = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/train"
    train_ann = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/train/_classifications.coco.json"
    
    val_root = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/valid"
    val_ann = "/root/autodl-tmp/seat_dataset/feichengdu_dataset/valid/_classifications.coco.json"