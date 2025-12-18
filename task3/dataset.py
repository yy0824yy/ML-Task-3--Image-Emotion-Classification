# dataset.py (不需要 csv 版)
import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, datasets
from config import Config

# 定义图像预处理和增强
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

val_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.CenterCrop(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders():
    # 分别构建带不同 transform 的数据集
    ds_train_base = datasets.ImageFolder(root=Config.TRAIN_IMAGE_DIR, transform=train_transforms)
    ds_val_base = datasets.ImageFolder(root=Config.TRAIN_IMAGE_DIR, transform=val_transforms)

    # 打印一下类别，确认路径
    print(f"Found classes: {ds_train_base.classes}")

    # 使用同一随机划分索引，确保两者对应一致
    dataset_len = len(ds_train_base)
    indices = list(range(dataset_len))
    import random
    random.seed(Config.SEED)
    random.shuffle(indices)
    split = int(0.8 * dataset_len)
    train_indices = indices[:split]
    val_indices = indices[split:]

    # 创建子集，子集将使用各自 base 的 transform
    train_dataset = Subset(ds_train_base, train_indices)
    val_dataset = Subset(ds_val_base, val_indices)

    # 计算加权采样器（根据训练子集的类别分布）
    targets = [ds_train_base.targets[i] for i in train_indices]
    num_classes = len(ds_train_base.classes)
    class_counts = [0] * num_classes
    for t in targets:
        class_counts[t] += 1
    total = sum(class_counts)
    class_weights = [total / (c + 1e-6) for c in class_counts]
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader

class TestImageDataset(Dataset):
    def __init__(self, file_names, img_dir, transform):
        self.file_names = file_names
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, fname

def get_test_loader():
    # 优先使用 sample_submission.csv 的顺序；若不存在则从测试目录生成按名字排序的列表
    file_names = None
    try:
        if os.path.exists(Config.SAMPLE_SUBMISSION):
            df = pd.read_csv(Config.SAMPLE_SUBMISSION)
            # 兼容列名：ID 或 filename 或第一列
            for col in ['ID', 'filename']:
                if col in df.columns:
                    file_names = df[col].tolist()
                    break
            if file_names is None:
                file_names = df.iloc[:, 0].tolist()
    except Exception:
        file_names = None

    if file_names is None:
        # 回退：直接从测试目录读取所有图片名，并排序保证确定性
        file_names = [f for f in os.listdir(Config.TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_names.sort()

    test_dataset = TestImageDataset(file_names, Config.TEST_IMAGE_DIR, val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    return test_loader

def get_test_loader_with_transform(transform):
    # 逻辑同 get_test_loader，但允许自定义 transform (用于 TTA)
    file_names = None
    try:
        if os.path.exists(Config.SAMPLE_SUBMISSION):
            df = pd.read_csv(Config.SAMPLE_SUBMISSION)
            for col in ['ID', 'filename']:
                if col in df.columns:
                    file_names = df[col].tolist()
                    break
            if file_names is None:
                file_names = df.iloc[:, 0].tolist()
    except Exception:
        file_names = None

    if file_names is None:
        file_names = [f for f in os.listdir(Config.TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_names.sort()

    test_dataset = TestImageDataset(file_names, Config.TEST_IMAGE_DIR, transform)
    # TTA 时 batch_size 可能需要减小，因为 crop 增加了数据量 (B, 10, C, H, W)
    # 但这里我们保持 batch_size，在 inference 中处理
    # 显存优化：强制将 batch_size 减半，防止 OOM
    safe_batch_size = max(1, Config.BATCH_SIZE // 4)
    test_loader = DataLoader(test_dataset, batch_size=safe_batch_size, shuffle=False, num_workers=2)
    return test_loader