# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from config import Config
from dataset import get_data_loaders
from model import get_emotion_model

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def mixup_criterion(criterion, pred, targets, lam):
    y_a, y_b = targets
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch, total_epochs, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training")
    for images, labels in loop:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero_grad(set_to_none=True)
        # 仅在前60%轮次启用 MixUp
        apply_mix = (epoch < int(0.6 * total_epochs)) and mixup_alpha > 0
        if apply_mix:
            mixed_images, targets, lam = mixup_data(images, labels, alpha=mixup_alpha)
            images = mixed_images
        
        with autocast(enabled=True):
            outputs = model(images)
            if apply_mix:
                y_a, y_b = targets
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # OneCycleLR 按 batch 更新
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    epoch_acc = 100 * correct / total
    return running_loss / len(loader), epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    return running_loss / len(loader), val_acc

def main():
    # 1. 准备数据
    train_loader, val_loader = get_data_loaders()
    
    # 2. 准备模型、损失函数、优化器
    model = get_emotion_model()
    # Label smoothing to improve generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    # OneCycleLR 调度器（更快更稳）
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=100.0
    )
    scaler = GradScaler(enabled=True)
    
    best_acc = 0.0
    
    # 3. 循环训练
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch, Config.EPOCHS, mixup_alpha=0.2)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Model Saved! (Best Val Acc: {best_acc:.2f}%)")

if __name__ == "__main__":
    main()