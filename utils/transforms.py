from torchvision.transforms import v2


def get_default_transform(is_train=True, size=(224, 224)):
    base_transform = v2.Compose([
        v2.Resize(size), 
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if is_train:
        return v2.Compose([
            # v2.RandomResizedCrop(size, scale=(0.8, 1.0)),  # 随机裁剪并缩放
            v2.RandomHorizontalFlip(),  # 水平翻转
            v2.RandomVerticalFlip(),  # 垂直翻转
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色扰动
            # v2.RandomRotation(degrees=15),  # 随机旋转
            # v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),  # 仿射变换
            base_transform
        ])
    else:
        # 验证模式：只使用基础变换
        return base_transform
