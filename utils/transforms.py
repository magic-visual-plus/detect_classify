from torchvision import transforms


def get_default_transform(is_train=True):
    compile = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.RandomVerticalFlip(),  # 垂直翻转
            compile
        ])
    else:
        return compile

