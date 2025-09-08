from torchvision.transforms import v2


def get_default_transform(is_train=True):
    base_transform = v2.Compose([
        v2.Resize((224, 224)), 
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if is_train:
        # 训练模式：添加数据增强
        return v2.Compose([
            v2.RandomHorizontalFlip(),  # 水平翻转
            v2.RandomVerticalFlip(),  # 垂直翻转
            base_transform
        ])
    else:
        # 验证模式：只使用基础变换
        return base_transform
