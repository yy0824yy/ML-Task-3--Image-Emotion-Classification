# model.py
import torch
import torch.nn as nn
from torchvision import models
from config import Config

class CBAMHead(nn.Module):
    def __init__(self, in_dim, hidden=512, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

def get_emotion_model():
    bb = Config.MODEL_BACKBONE.lower()
    if bb == "resnet101":
        print("Loading Pretrained ResNet101 Model...")
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        in_dim = model.fc.in_features
    elif bb == "convnext_t":
        print("Loading Pretrained ConvNeXt-Tiny Model...")
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_dim = model.classifier[2].in_features
    elif bb == "convnext_s":
        print("Loading Pretrained ConvNeXt-Small Model...")
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        in_dim = model.classifier[2].in_features
    elif bb == "convnext_b":
        print("Loading Pretrained ConvNeXt-Base Model...")
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        in_dim = model.classifier[2].in_features
    elif bb == "efficientnet_v2_s":
        print("Loading Pretrained EfficientNetV2-S Model...")
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_dim = model.classifier[1].in_features
    elif bb == "efficientnet_v2_m":
        print("Loading Pretrained EfficientNetV2-M Model...")
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        in_dim = model.classifier[1].in_features
    elif bb == "swin_s":
        print("Loading Pretrained Swin-S Model...")
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        in_dim = model.head.in_features
    else:
        print("Loading Pretrained ResNet50 Model...")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_dim = model.fc.in_features

    head = CBAMHead(in_dim, hidden=512, num_classes=Config.NUM_CLASSES) if Config.USE_CBAM else nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, Config.NUM_CLASSES)
    )

    if bb.startswith("resnet"):
        model.fc = head
    elif bb.startswith("efficientnet"):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            head
        )
    elif bb.startswith("swin"):
        model.head = head
    else:
        # ConvNeXt: classifier = [LayerNorm, Linear, GELU, Linear]
        # model.classifier[-1] = nn.Linear(in_dim, Config.NUM_CLASSES)
        # 将前面的分类器替换为我们自定义 head（简化为线性头以保持稳定）
        model.classifier = nn.Sequential(
            nn.Flatten(),
            head
        )

    return model.to(Config.DEVICE)