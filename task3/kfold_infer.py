import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

from config import Config
from dataset import get_test_loader_with_transform
from model import get_emotion_model

def infer_ensemble(weight_paths):
    # Define Multi-Scale TTA Transforms
    # Scale 1: Tighter crop (235 -> 224)
    # Scale 2: Standard (256 -> 224)
    # Scale 3: Looser crop (280 -> 224)
    scales = [235, 256, 280]
    tta_transforms = []
    for s in scales:
        t = transforms.Compose([
            transforms.Resize((s, s)),
            transforms.TenCrop(Config.IMAGE_SIZE),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])
        tta_transforms.append(t)

    models = []
    print(f"Found {len(weight_paths)} weights: {weight_paths}")
    
    # 定义不同模型的权重
    # 逻辑：大模型 > 小模型，新架构 > 旧架构
    weights_map = {
        "swin_s": 0.5,           # 降低 Swin 权重，防止其性能不佳拖累整体
        "convnext_b": 1.5,       # 最强模型
        "efficientnet_v2_m": 1.4,
        "efficientnet_v2_s": 1.2,
        "convnext_s": 1.2,
        "convnext_t": 1.0,
        "resnet101": 0.8,        # 相对较弱
        "resnet50": 0.6
    }

    for p in weight_paths:
        # Parse backbone from filename
        filename = os.path.basename(p)
        backbone = "resnet101" # Default
        
        if "_fold" in filename:
            parts = filename.split("_fold")
            prefix = parts[0] # "best_model" or "best_model_convnext_t"
            if len(prefix) > 10: # len("best_model") == 10
                backbone = prefix.replace("best_model_", "")
        
        # 获取该模型的权重
        model_weight = weights_map.get(backbone, 1.0)
        print(f"Loading {backbone} from {filename} (Weight: {model_weight})...")
        
        # Dynamically set backbone for model creation
        original_backbone = Config.MODEL_BACKBONE
        Config.MODEL_BACKBONE = backbone
        try:
            m = get_emotion_model()
            state = torch.load(p, map_location=Config.DEVICE)
            m.load_state_dict(state)
            m.eval()
            # 存储模型及其权重
            models.append((m, model_weight))
        except Exception as e:
            print(f"Error loading {p}: {e}")
        finally:
            Config.MODEL_BACKBONE = original_backbone

    if not models:
        print("No models loaded!")
        return

    # Multi-Scale Inference Loop
    final_preds = []
    final_fnames = []
    
    # Loaders for each scale
    loaders = [get_test_loader_with_transform(t) for t in tta_transforms]
    
    # Mapping info
    print("Assuming Alphabetical Class Mapping: 0:Angry, 1:Fear, 2:Happy, 3:Neutral, 4:Sad, 5:Surprise")
    print(f"Starting Multi-Scale Inference (Scales: {scales})...")

    with torch.no_grad():
        # Iterate through the first loader to get batches, others are synced by index
        # Note: This assumes deterministic ordering (which we ensured in dataset.py)
        for (batch_s1, batch_s2, batch_s3) in tqdm(zip(loaders[0], loaders[1], loaders[2]), total=len(loaders[0]), desc='Infer Multi-Scale'):
            imgs_list = [batch_s1[0], batch_s2[0], batch_s3[0]]
            names = batch_s1[1] # Names should be identical
            
            batch_probs_sum = None
            
            for img_scale in imgs_list:
                # img_scale: (B, 10, C, H, W)
                bs, n_crops, c, h, w = img_scale.size()
                inputs = img_scale.view(-1, c, h, w).to(Config.DEVICE)
                
                scale_logits = None
                total_weight = 0.0
                
                for m, w_val in models:
                    out = m(inputs) # (B*10, NumClasses)
                    # 回退到 Logit 平均：实验证明 Logit 平均在同类任务中往往更鲁棒
                    if scale_logits is None:
                        scale_logits = out * w_val
                    else:
                        scale_logits += out * w_val
                    total_weight += w_val
                
                # Average over models for this scale
                scale_logits = scale_logits / total_weight
                
                # Accumulate across scales
                if batch_probs_sum is None:
                    batch_probs_sum = scale_logits
                else:
                    batch_probs_sum += scale_logits
                
                # 显存清理
                del inputs, out
                torch.cuda.empty_cache()
            
            # Average over scales
            final_batch_logits = batch_probs_sum / len(scales)
            
            # Reshape back to (B, 10, NumClasses)
            final_batch_logits = final_batch_logits.view(bs, n_crops, -1)
            
            # Average over crops (Mean Logits)
            final_logits = torch.mean(final_batch_logits, dim=1)
            
            # Final Softmax
            final_probs = torch.softmax(final_logits, dim=1)
            
            pred = torch.argmax(final_probs, dim=1)
            final_preds.extend(pred.detach().cpu().tolist())
            final_fnames.extend(names)
            
            # 显存清理
            del batch_probs_sum, final_batch_logits, final_logits, final_probs
            torch.cuda.empty_cache()

    sub = pd.DataFrame({'ID': [str(x) for x in final_fnames], 'Emotion': [int(x) for x in final_preds]})
    
    # Apply Kaggle Target Mapping (Fix for 0.731 score)
    # Local (Alpha): 0:Angry, 1:Fear, 2:Happy, 3:Neutral, 4:Sad, 5:Surprise
    # Kaggle Target: 0:Angry, 1:Fear, 2:Happy, 3:Sad, 4:Surprise, 5:Neutral
    mapping = {
        0: 0, # Angry -> Angry
        1: 1, # Fear -> Fear
        2: 2, # Happy -> Happy
        3: 5, # Neutral -> Neutral
        4: 3, # Sad -> Sad
        5: 4  # Surprise -> Surprise
    }
    sub['Emotion'] = sub['Emotion'].map(mapping)
    print("Applied Class Mapping: Neutral(3)->5, Sad(4)->3, Surprise(5)->4")

    sub.to_csv(Config.RESULT_CSV, index=False, encoding='utf-8', lineterminator='\n')
    print(f"Saved ensemble submission to {Config.RESULT_CSV}")

import glob

def main():
    # Find all best_model*.pth files
    weight_paths = glob.glob("./best_model*.pth")
    if not weight_paths:
        print("No fold weights found. Please run kfold_train.py first.")
        return
    infer_ensemble(weight_paths)

if __name__ == '__main__':
    main()
