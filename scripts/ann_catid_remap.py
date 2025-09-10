import os
import json
import argparse
import copy

def remap_coco_categories(coco_path, output_path, model_id2name):
    """
    将coco标注文件的categories和annotations中的类别id映射到模型的id，并补全缺失的类别
    :param coco_path: 原始coco json文件路径
    :param output_path: 修正后的coco json文件保存路径
    :param model_id2name: 模型的id2name字典，例如 {0: "cat", 1: "dog", ...}
    """
    # 读取coco文件
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1. 构建coco原始类别name到id的映射
    coco_name2id = {cat['name']: cat['id'] for cat in coco['categories']}
    coco_id2name = {cat['id']: cat['name'] for cat in coco['categories']}

    # 2. 构建模型类别name到id的映射
    model_name2id = {v: k for k, v in model_id2name.items()}

    # 3. 构建coco原始id到模型id的映射（只映射有交集的类别）
    catid_map = {}
    for coco_id, name in coco_id2name.items():
        if name in model_name2id:
            catid_map[coco_id] = model_name2id[name]
        else:
            print(f"警告: coco类别'{name}'未在模型类别中找到，将被忽略。")

    # 4. 修正annotations中的category_id
    new_annotations = []
    for ann in coco['annotations']:
        old_cat = ann['category_id']
        if old_cat in catid_map:
            ann_new = copy.deepcopy(ann)
            ann_new['category_id'] = catid_map[old_cat]
            new_annotations.append(ann_new)
        else:
            print(f"警告: annotation中category_id={old_cat}未能映射，将被忽略。")

    # 5. 构建新的categories，补全缺失的类别
    new_categories = []
    for model_id, name in model_id2name.items():
        new_categories.append({
            "id": model_id,
            "name": name,
            "supercategory": "none"
        })

    # 6. 组装新coco字典
    new_coco = copy.deepcopy(coco)
    new_coco['categories'] = new_categories
    new_coco['annotations'] = new_annotations

    # 7. 保存新coco文件
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)
    print(f"修正后的coco文件已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("COCO类别id修正脚本")
    parser.add_argument("--input", type=str, required=True, help="原始coco json文件路径")
    parser.add_argument("--output", type=str, required=True, help="修正后coco json文件保存路径")
    args = parser.parse_args()
    id2name = {
        0: '其他', 1: '划伤', 2: '压痕', 
        3: '吊紧', 4: '异物外漏', 5: '折痕', 6: '抛线',
        7: '拼接间隙', 8: '水渍', 9: '烫伤', 10: '破损', 
        11: '碰伤', 12: '红标签', 13: '线头', 14: '脏污', 
        15: '褶皱(T型)', 16: '褶皱（重度）', 17: '重跳针'
    }

    remap_coco_categories(args.input, args.output, id2name)
