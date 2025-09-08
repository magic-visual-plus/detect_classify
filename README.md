# README

## install 

### 1. 安装 DINOv3

```shell
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3
pip install -e .
```

### 2. 

### 3. 安装其它依赖

```shell
pip install opencv-python
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install lightning
```

## 快速开始

### 1. 配置设置

项目使用YAML配置文件进行参数管理。默认配置文件位于 `configs/default.yaml`：

```yaml
lr: 1e-4
max_epochs: 30
batch_size: 64
num_workers: 12
dataset:
  train_path: "/path/to/train/data"
  val_path: "/path/to/val/data"
  train_ann: "/path/to/train/annotations.json"
  val_ann: "/path/to/val/annotations.json"
min_bbox_area: 400
backbone_weights: "/path/to/pretrained/weights.pth"
```

### 2. 训练模型

```shell
# 使用默认配置训练
python trainer.py --config configs/default.yaml

# 覆盖特定参数
python trainer.py --config configs/default.yaml --batch_size 32 --lr 5e-5

# 使用测试配置
python trainer.py --config configs/test.yaml
```


### 3. 数据集处理
    基于一阶段DINO目标检测模型生成数据集
```shell
python make/stage1_predict_dataset.py /path/to/model /path/to/dataset
```

## 配置说明

### 训练参数

- `lr`: 学习率 (默认: 1e-4)
- `max_epochs`: 最大训练轮数 (默认: 30)
- `batch_size`: 批次大小 (默认: 64)
- `num_workers`: 数据加载线程数 (默认: 12)

### 数据集参数

- `train_path`: 训练数据路径
- `val_path`: 验证数据路径
- `train_ann`: 训练标注文件路径
- `val_ann`: 验证标注文件路径
- `min_bbox_area`: 最小边界框面积阈值

### 模型参数

- `backbone_weights`: 预训练权重路径
- `freeze_backbone`: 是否冻结骨干网络 (默认: True)
