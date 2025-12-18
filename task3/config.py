# config.py
import torch
import os

class Config:
    # ================= 路径设置 (关键修改) =================
    # ".." 表示返回上一级目录，进入 fer_data 文件夹
    TRAIN_IMAGE_DIR = "../fer_data/train"       
    TEST_IMAGE_DIR = "../fer_data/test"         
    
    # 假设你的 CSV 文件也在 fer_data 文件夹里
    # 如果 CSV 文件是在 Task3 根目录下（而不是 fer_data 里），请去掉中间的 fer_data/，改为 "../train.csv"
    TRAIN_CSV = "../fer_data/train.csv"         
    SAMPLE_SUBMISSION = "../submission.csv" 
    
    # 结果保存路径（保存在当前代码文件夹下即可）
    MODEL_SAVE_PATH = "./best_model.pth" 
    RESULT_CSV = "../submission.csv"      
    
    # ================= 图像与模型参数 =================
    IMAGE_SIZE = 224    # 模型标准输入大小
    MODEL_BACKBONE = "convnext_b"  # 可选: resnet50/101, convnext_t/s/b, efficientnet_v2_s/m, swin_s
    USE_CBAM = True              # 是否在分类头加入轻量注意力（CBAM简化版）
    USE_FOCAL_LOSS = False       # 关闭 Focal Loss，回退到 Label Smoothing
    NUM_CLASSES = 6     # 你的任务为 6 类
    
    # ================= 训练超参数 =================
    BATCH_SIZE = 32     # 如显存吃紧可降到 16
    EPOCHS = 25         # 增加训练轮数，配合 OneCycleLR
    LEARNING_RATE = 3e-4
    
    # 自动检测设备 (GPU/CPU)
    # 指定使用 0 号 GPU (第一个)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 随机种子 (保证结果可复现)
    SEED = 42

print(f"Using Device: {Config.DEVICE}")
# 简单的路径检查，防止运行报错
if not os.path.exists(Config.TRAIN_IMAGE_DIR):
    print(f"⚠️ 警告: 找不到训练图片路径: {Config.TRAIN_IMAGE_DIR}")
    print("请检查你的文件夹结构是否为：Task3 -> fer_data -> train")