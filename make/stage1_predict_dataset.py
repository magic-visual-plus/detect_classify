import sys
import os
import json
import cv2
from tqdm import tqdm
from collections import Counter
import pandas as pd
from hq_det.models.dino import hq_dino


def load_coco_data(path):
    with open(path, "r") as f:
        coco_data = json.load(f)
    return coco_data


def replace_coco_file(coco_file, id2names):
    """
    coco文件的类别id为模型预测的类别id
    Args:
        coco_file: coco文件路径
        id2names: 类别id到名称的映射
    Returns:
        coco_data: 替换后的coco数据
    """
    coco_data = load_coco_data(coco_file)
    new_categories = [
        {
            "id": i,
            "name": name,
            "supercategory": "none"
        }
        for i, name in enumerate(id2names.values())
    ]
    for item in reversed(new_categories):
        name = item['name']
        id = item['id']
        raw_id_list = [i for i, c in enumerate(coco_data['categories']) if c['name'] == name]
        if not raw_id_list:
            print(f"not find label: {name}")
            continue 
        raw_id = raw_id_list[0]
        for ann in coco_data['annotations']:
            if ann['category_id'] == raw_id:
                ann['category_id'] = id
    for ann in coco_data['annotations']:
        ann['source'] = 'gt'
    coco_data['categories'] = new_categories

    return coco_data


def check_coco_category_counts(coco):
    """
    检查COCO文件中每个类别的标签数量。

    参数:
        coco: 可以是coco字典对象，或coco文件路径（.json）
        show: 是否打印统计信息

    返回:
        result: dict，key为类别id，value为(类别名, 标签数量)
    """
    if isinstance(coco, str) and os.path.isfile(coco):
        with open(coco, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    else:
        coco_data = coco

    category_counts = Counter([ann['category_id'] for ann in coco_data['annotations']])
    result = {}
    for cat in coco_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        count = category_counts.get(cat_id, 0)
        result[cat_id] = (cat_name, count)
    return result

def predict_coco(coco_file, model, input_path, confidence=0.3, max_size=1536):
    """
    预测coco文件
    Args:
        coco_file: coco文件路径
        model: 模型
        input_path: 输入路径
        output_path: 输出路径
    """
    coco_data = replace_coco_file(coco_file, model.id2names)
    coco_output = coco_data.copy()

    annotation_id = 1
    for image_info in tqdm(coco_data['images']):
        img_path = os.path.join(input_path, image_info['file_name'])
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        assert height == image_info['height'] and width == image_info['width']
        result = model.predict([img], bgr=True, confidence=confidence, max_size=max_size)[0]
        bboxes = result.bboxes
        scores = result.scores
        labels = result.cls
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_info['id'],
                "category_id": int(labels[i]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(scores[i]),
                "area": float(w * h),
                "iscrowd": 0,
                "source": "predict"
            })
            annotation_id += 1

    return coco_output

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
    import numpy as np

    # 获取当前最大类别id
    if len(coco_data['categories']) == 0:
        max_cat_id = 0
    else:
        max_cat_id = max(cat['id'] for cat in coco_data['categories'])
    background_id = max_cat_id + 1

    # 检查是否已存在背景类别，若不存在则添加
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

def merge_coco_categories(coco_data, merge_dict):
    """
    合并COCO标签类别

    参数:
        coco_data: COCO格式的字典数据
        merge_dict: dict，key为合并后的新类别名，value为要合并的原类别名列表

    返回:
        合并后的coco_data
    """
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
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    coco_file = os.path.join(input_path, "_annotations.coco.json")
    model = hq_dino.HQDINO(model=model_path)
    model.eval()
    model.to("cuda:0")
    coco_output = predict_coco(coco_file, model, input_path, confidence=0.3, max_size=1536)
    # with open("predict_coco.json", "w", encoding="utf-8") as f:
    #     json.dump(coco_output, f, ensure_ascii=False, indent=2)
    coco_output = correct_predicted_categories(coco_output)
    # coco_output = merge_coco_categories(
    #     coco_output,
    #     merge_dict={
    #         '缺陷': model.id2names.values(),
    #         "背景": ["背景"]
    #     }
    # )
    with open(os.path.join(input_path, "_classification.coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)