import os
import json
import random
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

def load_coco_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_coco_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_coco_dataset(
    train_img_dir, train_ann_path, 
    val_img_dir, val_ann_path, 
    out_train_dir, out_val_dir, 
    out_train_ann, out_val_ann, 
    train_ratio=0.8, random_seed=42
):
    # 加载原始coco标注
    print("加载训练集标注文件:", train_ann_path)
    train_coco = load_coco_json(train_ann_path)
    print("加载验证集标注文件:", val_ann_path)
    val_coco = load_coco_json(val_ann_path)

    # 合并图片和标注，重新分配ID避免冲突
    print(f"原始训练集图片数: {len(train_coco['images'])}")
    print(f"原始验证集图片数: {len(val_coco['images'])}")
    print(f"原始训练集标注数: {len(train_coco['annotations'])}")
    print(f"原始验证集标注数: {len(val_coco['annotations'])}")
    
    # 获取最大ID用于重新分配
    max_img_id = max(max(img["id"] for img in train_coco["images"]), 
                     max(img["id"] for img in val_coco["images"]))
    max_ann_id = max(max(ann["id"] for ann in train_coco["annotations"]), 
                     max(ann["id"] for ann in val_coco["annotations"]))
    
    # 重新分配验证集的图片ID和标注ID
    val_img_id_offset = max_img_id + 1
    val_ann_id_offset = max_ann_id + 1
    
    # 创建ID映射和文件名映射
    val_img_id_map = {}
    val_ann_id_map = {}
    val_img_filename_map = {}  # 新ID到文件名的映射
    
    # 重新分配验证集图片ID
    for img in val_coco["images"]:
        old_id = img["id"]
        new_id = old_id + val_img_id_offset
        val_img_id_map[old_id] = new_id
        val_img_filename_map[new_id] = img["file_name"]  # 保存文件名映射
        img["id"] = new_id
    
    # 重新分配验证集标注ID和对应的图片ID
    for ann in val_coco["annotations"]:
        old_ann_id = ann["id"]
        old_img_id = ann["image_id"]
        new_ann_id = old_ann_id + val_ann_id_offset
        new_img_id = val_img_id_map[old_img_id]
        
        val_ann_id_map[old_ann_id] = new_ann_id
        ann["id"] = new_ann_id
        ann["image_id"] = new_img_id
    
    # 合并图片和标注
    all_images = train_coco["images"] + val_coco["images"]
    all_annotations = train_coco["annotations"] + val_coco["annotations"]
    categories = train_coco["categories"]  # 假设类别一致
    
    print(f"合并后图片数: {len(all_images)}")
    print(f"合并后标注数: {len(all_annotations)}")
    print(f"验证集图片ID偏移: {val_img_id_offset}")
    print(f"验证集标注ID偏移: {val_ann_id_offset}")

    # 按image_id分组标注
    imgid2anns = defaultdict(list)
    for ann in all_annotations:
        imgid2anns[ann["image_id"]].append(ann)

    # 随机划分图片
    random.seed(random_seed)
    all_image_ids = [img["id"] for img in all_images]
    random.shuffle(all_image_ids)
    train_num = int(len(all_image_ids) * train_ratio)
    train_img_ids = set(all_image_ids[:train_num])
    val_img_ids = set(all_image_ids[train_num:])

    # 分别收集图片和标注
    train_images = [img for img in all_images if img["id"] in train_img_ids]
    val_images = [img for img in all_images if img["id"] in val_img_ids]
    train_anns = [ann for img_id in train_img_ids for ann in imgid2anns[img_id]]
    val_anns = [ann for img_id in val_img_ids for ann in imgid2anns[img_id]]

    # 构建新的coco字典
    train_coco_new = {
        "images": train_images,
        "annotations": train_anns,
        "categories": categories
    }
    val_coco_new = {
        "images": val_images,
        "annotations": val_anns,
        "categories": categories
    }

    # 创建输出文件夹
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)

    # 拷贝图片到新目录
    print("拷贝训练集图片...")
    for img in tqdm(train_images):
        img_name = img["file_name"]
        # 训练集图片优先从训练集目录找，找不到再从验证集目录找
        src_path = os.path.join(train_img_dir, img_name)
        if not os.path.exists(src_path):
            src_path = os.path.join(val_img_dir, img_name)
        dst_path = os.path.join(out_train_dir, img_name)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
    
    print("拷贝验证集图片...")
    for img in tqdm(val_images):
        img_name = img["file_name"]
        # 验证集图片优先从验证集目录找，找不到再从训练集目录找
        src_path = os.path.join(val_img_dir, img_name)
        if not os.path.exists(src_path):
            src_path = os.path.join(train_img_dir, img_name)
        dst_path = os.path.join(out_val_dir, img_name)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

    # 保存新的coco标注
    print("保存新的训练集标注到:", out_train_ann)
    save_coco_json(train_coco_new, out_train_ann)
    print("保存新的验证集标注到:", out_val_ann)
    save_coco_json(val_coco_new, out_val_ann)
    print("划分完成！")
    print(f"训练集图片数: {len(train_images)}, 验证集图片数: {len(val_images)}")
    print(f"训练集标注数: {len(train_anns)}, 验证集标注数: {len(val_anns)}")
    
    # 验证总数不变
    original_img_count = len(all_images)
    new_img_count = len(train_images) + len(val_images)
    original_ann_count = len(all_annotations)
    new_ann_count = len(train_anns) + len(val_anns)
    
    print(f"\n验证结果:")
    print(f"原始图片总数: {original_img_count}, 新图片总数: {new_img_count}, 是否相等: {original_img_count == new_img_count}")
    print(f"原始标注总数: {original_ann_count}, 新标注总数: {new_ann_count}, 是否相等: {original_ann_count == new_ann_count}")
    
    if original_img_count != new_img_count or original_ann_count != new_ann_count:
        print("警告：总数不匹配！")
    else:
        print("✓ 总数验证通过！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO数据集随机重新划分脚本")
    parser.add_argument("--train_img_dir", type=str, required=True, help="原训练集图片文件夹")
    parser.add_argument("--val_img_dir", type=str, required=True, help="原验证集图片文件夹")
    parser.add_argument("--out_train_dir", type=str, required=True, help="输出训练集图片文件夹")
    parser.add_argument("--out_val_dir", type=str, required=True, help="输出验证集图片文件夹")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 生成输出coco标注文件路径，均为"_ann.coco.json"
    coco_file_name = "_annotations.coco.json"
    out_train_ann = os.path.join(args.out_train_dir, coco_file_name)
    out_val_ann = os.path.join(args.out_val_dir, coco_file_name)

    # train_ann和val_ann也按输入图片目录下的_ann.coco.json寻找
    train_ann = os.path.join(args.train_img_dir, coco_file_name)
    val_ann = os.path.join(args.val_img_dir, coco_file_name)

    split_coco_dataset(
        args.train_img_dir, train_ann,
        args.val_img_dir, val_ann,
        args.out_train_dir, args.out_val_dir,
        out_train_ann, out_val_ann,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )
