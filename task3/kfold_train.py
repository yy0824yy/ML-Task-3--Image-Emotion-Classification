import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset
from PIL import Image
import pandas as pd

from config import Config
from model import get_emotion_model

# 定义伪标签数据集类
class PseudoDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        label = int(row['label'])
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.15)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_base_datasets():
    ds_train_base = datasets.ImageFolder(root=Config.TRAIN_IMAGE_DIR, transform=transforms.Compose([
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]))
    ds_val_base = datasets.ImageFolder(root=Config.TRAIN_IMAGE_DIR, transform=transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]))
    return ds_train_base, ds_val_base


def train_fold(train_idx, val_idx, fold):
    ds_train_base, ds_val_base = get_base_datasets()
    train_ds = Subset(ds_train_base, train_idx)
    val_ds = Subset(ds_val_base, val_idx)

    # 检查并加载伪标签数据
    pseudo_csv = "pseudo_labels.csv"
    if os.path.exists(pseudo_csv):
        print(f"[Fold {fold}] Loading Pseudo Labels from {pseudo_csv}...")
        # 伪标签数据使用与训练集相同的增强策略
        pseudo_ds = PseudoDataset(pseudo_csv, Config.TEST_IMAGE_DIR, ds_train_base.transform)
        # 将伪标签数据合并到训练集中
        train_ds = ConcatDataset([train_ds, pseudo_ds])
        print(f"[Fold {fold}] Added {len(pseudo_ds)} pseudo-labeled samples to training set.")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = get_emotion_model()
    if getattr(Config, 'USE_FOCAL_LOSS', False):
        print(f"[Fold {fold}] Using Focal Loss")
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=100.0
    )
    scaler = GradScaler(enabled=True)

    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                with autocast(enabled=True):
                    outputs = model(images)
                    _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            save_name = f"./best_model_{Config.MODEL_BACKBONE}_fold{fold}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"[Fold {fold}] Model Saved to {save_name}! (Best Val Acc: {best_acc:.2f}%)")


def main():
    print(f"Using Device: {Config.DEVICE}")
    # Prepare indices and labels for stratified split
    base = datasets.ImageFolder(root=Config.TRAIN_IMAGE_DIR)
    labels = base.targets
    indices = list(range(len(base)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        print(f"\n=== Training Fold {fold}/5 ===")
        train_fold(train_idx, val_idx, fold)


if __name__ == '__main__':
    main()
