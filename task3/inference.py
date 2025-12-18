import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from config import Config
from dataset import get_test_loader_with_transform
from model import get_emotion_model

def predict():
    # Define TTA Transform (TenCrop)
    resize_size = 256
    tta_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.TenCrop(Config.IMAGE_SIZE),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
    
    # 1. 加载测试数据
    test_loader = get_test_loader_with_transform(tta_transform)
    
    # 2. 加载模型结构
    model = get_emotion_model()
    
    # 3. 加载训练好的权重
    print(f"Loading weights from {Config.MODEL_SAVE_PATH}")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model file {Config.MODEL_SAVE_PATH} not found!")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    model.eval()
    
    predictions = []
    file_names = []
    
    print("Starting Prediction (TenCrop)...")
    print("Assuming Alphabetical Class Mapping: 0:Angry, 1:Fear, 2:Happy, 3:Neutral, 4:Sad, 5:Surprise")

    with torch.no_grad():
        for images, fnames in tqdm(test_loader):
            # images: (B, 10, C, H, W)
            bs, n_crops, c, h, w = images.size()
            inputs = images.view(-1, c, h, w).to(Config.DEVICE)
            
            # Forward
            out = model(inputs) # (B*10, NumClasses)
            
            # Reshape and Average
            out = out.view(bs, n_crops, -1)
            out = torch.mean(out, dim=1) # (B, NumClasses)
            
            _, predicted = torch.max(out, 1)
            predictions.extend([int(p) for p in predicted.cpu().tolist()])
            file_names.extend(fnames)
            
    # 4. 生成提交文件
    submission = pd.DataFrame({
        "ID": [str(x) for x in file_names],
        "Emotion": [int(x) for x in predictions]
    })
    
    submission.to_csv(Config.RESULT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Submission file saved to {Config.RESULT_CSV}")

if __name__ == "__main__":
    predict()
