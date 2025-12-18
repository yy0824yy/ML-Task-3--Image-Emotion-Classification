import os
import torch
import pandas as pd
import glob
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import Config
from model import get_emotion_model

# 定义一个简单的 Dataset 用于加载测试图片
class PseudoLabelDataset(Dataset):
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

def generate_pseudo_labels():
    # 1. 准备 TTA Transform (TenCrop)
    resize_size = 256
    tta_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.TenCrop(Config.IMAGE_SIZE),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])

    # 2. 加载所有模型
    weight_paths = glob.glob("./best_model*.pth")
    if not weight_paths:
        print("No models found!")
        return

    models = []
    print(f"Loading {len(weight_paths)} models for pseudo-label generation...")
    
    # 定义权重 (与 kfold_infer.py 保持一致)
    weights_map = {
        "swin_s": 0.5,
        "convnext_b": 1.5,
        "efficientnet_v2_m": 1.4,
        "efficientnet_v2_s": 1.2,
        "convnext_s": 1.2,
        "convnext_t": 1.0,
        "resnet101": 0.8,
        "resnet50": 0.6
    }
    
    for p in weight_paths:
        filename = os.path.basename(p)
        backbone = "resnet101" # Default
        if "_fold" in filename:
            parts = filename.split("_fold")
            prefix = parts[0]
            if len(prefix) > 10:
                backbone = prefix.replace("best_model_", "")
        
        model_weight = weights_map.get(backbone, 1.0)
        print(f"Loading {backbone} (Weight: {model_weight})...")

        # 动态切换 backbone 加载模型
        original_backbone = Config.MODEL_BACKBONE
        Config.MODEL_BACKBONE = backbone
        try:
            m = get_emotion_model()
            state = torch.load(p, map_location=Config.DEVICE)
            m.load_state_dict(state)
            m.eval()
            models.append((m, model_weight))
        except Exception as e:
            print(f"Error loading {p}: {e}")
        finally:
            Config.MODEL_BACKBONE = original_backbone

    # 3. 准备数据加载器
    file_names = [f for f in os.listdir(Config.TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    file_names.sort()
    dataset = PseudoLabelDataset(file_names, Config.TEST_IMAGE_DIR, tta_transform)
    # 显存优化：Batch Size 减半
    safe_batch_size = max(1, Config.BATCH_SIZE // 2)
    loader = DataLoader(dataset, batch_size=safe_batch_size, shuffle=False, num_workers=4)

    # 4. 推理并筛选高置信度样本
    high_conf_samples = []
    high_conf_labels = []
    
    CONFIDENCE_THRESHOLD = 0.95  # 只选择置信度 > 95% 的样本
    
    print(f"Starting inference to generate pseudo labels (Threshold: {CONFIDENCE_THRESHOLD})...")
    
    with torch.no_grad():
        for images, fnames in tqdm(loader):
            bs, n_crops, c, h, w = images.size()
            inputs = images.view(-1, c, h, w).to(Config.DEVICE)
            
            logits_sum = None
            total_weight = 0.0
            
            for m, w_val in models:
                out = m(inputs)
                # 回退到 Logit 平均
                if logits_sum is None:
                    logits_sum = out * w_val
                else:
                    logits_sum += out * w_val
                total_weight += w_val
            
            avg_logits = logits_sum / total_weight
            avg_logits = avg_logits.view(bs, n_crops, -1)
            final_logits = torch.mean(avg_logits, dim=1)
            
            # 计算 Softmax 概率
            probs = torch.softmax(final_logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            # 筛选
            for i in range(bs):
                if max_probs[i].item() > CONFIDENCE_THRESHOLD:
                    high_conf_samples.append(fnames[i])
                    high_conf_labels.append(preds[i].item())
            
            # 显存清理
            del inputs, logits_sum, avg_logits, final_logits, probs
            torch.cuda.empty_cache()

    # 5. 保存伪标签文件
    # 注意：这里保存的是模型原始输出的 Label (0-5 字母序)，直接用于训练即可，不需要映射
    df = pd.DataFrame({
        "filename": high_conf_samples,
        "label": high_conf_labels
    })
    
    output_path = "pseudo_labels.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} pseudo-labels out of {len(file_names)} test images.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_pseudo_labels()
