# 面部表情识别 (FER) - 任务 3 解决方案

本项目是针对面部表情识别任务的高分解决方案，在 Kaggle 评估集上取得了 **0.745** 的最终得分。项目采用了加权异构模型集成、伪标签半监督学习、多尺度 TTA 等先进技术。

## 🏆 最终成果
- **最佳分数**: 0.745
- **核心策略**: 加权异构集成 (Weighted Heterogeneous Ensemble) + 多尺度测试时增强 (Multi-Scale TTA) + 迭代伪标签 (Iterative Pseudo-Labeling)

## 🛠️ 技术方案概览

### 1. 模型架构 (Weighted Heterogeneous Ensemble)
为了捕捉多样化的特征，我们构建了一个包含 **20 个模型** 的异构集成系统：
- **ConvNeXt-Base** (权重 1.5): 主力模型，性能最强。
- **EfficientNetV2-M** (权重 1.4): 平衡精度与参数量。
- **EfficientNetV2-S** (权重 1.2): 高效轻量。
- **ConvNeXt-Tiny** (权重 1.0): 补充轻量级特征。
- **ResNet101** (权重 0.8): 传统的基准模型，提供稳健性。

**注意力机制**: 所有模型分类头前均引入了 **CBAM (Convolutional Block Attention Module)**，强化对人脸关键区域（如眼睛、嘴巴）的关注。

### 2. 训练策略
- **5-Fold 交叉验证**: 使用 StratifiedKFold 确保训练集和验证集分布一致。
- **优化器**: AdamW
- **学习率调度**: OneCycleLR (Max LR = 3e-4)
- **损失函数**: CrossEntropyLoss with Label Smoothing (0.15)
- **伪标签 (Pseudo-Labeling)**: 利用集成模型预测测试集，筛选置信度 > 0.95 的样本加入训练，实现半监督学习。

### 3. 推理策略 (Inference)
- **多尺度 TTA (Multi-Scale TTA)**: 对测试图像进行 3 个尺度的缩放 (235, 256, 280)，并在每个尺度下进行 TenCrop (10重裁剪)，共计 30 次推理取平均。
- **加权 Logit 融合**: 不同模型根据其验证集表现被赋予不同权重，最终结果由加权 Logit 平均得出。

## 📂 文件结构说明
```
task3/
├── config.py                 # 全局配置文件 (路径、超参数、模型选择)
├── dataset.py                # 数据加载、预处理与增强 (RandAugment)
├── model.py                  # 定义所有支持的模型架构 (及 CBAM 模块)
├── kfold_train.py            # 5折交叉验证训练脚本 (包含伪标签加载逻辑)
├── kfold_infer.py            # 加权集成推理脚本 (包含 Multi-Scale TTA)
├── generate_pseudo_labels.py # 生成伪标签脚本
├── generate_dist_chart.py    # (工具) 生成类别分布对比图
├── generate_score_chart.py   # (工具) 生成分数提升趋势图
├── check_distribution.py     # (工具) 检查训练集与预测结果的分布差异
└── README.md                 # 说明文档
```

## 🚀 快速开始

### 1. 环境配置
请确保安装以下依赖库：
```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm seaborn matplotlib
```

### 2. 数据准备
请确保数据放置在 `../fer_data` 目录下，结构如下：
```
../fer_data/
    train/
        Angry/
        Fear/
        ...
    test/
        image1.jpg
        ...
```

### 3. 运行步骤 (复现 0.745 分数)

**步骤 A: 训练基础模型 (分别修改 config.py 中的 BACKBONE 并运行)**
```bash
# 修改 config.py 将 MODEL_BACKBONE 设为 'convnext_b', 'efficientnet_v2_m' 等
python3 kfold_train.py
```
*(注意：完全复现需要训练 config.py 中列出的所有 backbone，总共约20个模型文件)*

**步骤 B: 生成伪标签 (可选，提升性能)**
利用已有模型生成高质量伪标签：
```bash
python3 generate_pseudo_labels.py
```
然后再次运行 `kfold_train.py` 进行微调训练。

**步骤 C: 最终推理**
执行加权集成与多尺度 TTA 推理，生成 `submission.csv`：
```bash
python3 kfold_infer.py
```

## 📈 优化历程
| 阶段 | 分数 | 关键改动 |
| :--- | :--- | :--- |
| **Baseline** | 0.731 | 修复标签映射错误 (Neutral/Sad Align) |
| **Ensemble** | 0.739 | 引入 ConvNeXt 和 EfficientNetV2 异构集成 |
| **Pseudo-Label** | 0.741 | 加入高置信度伪标签训练 |
| **Weighted** | 0.744 | 对强模型赋予更高投票权重 |
| **Final** | **0.745** | **Multi-Scale TTA + Logit Averaging + No Focal Loss** |

## ⚠️ 注意事项
1. **标签映射**: 本地训练数据按字母序排列 (0:Angry, 3:Neutral, 4:Sad, 5:Surprise)，而 Kaggle 评测标准不同。`kfold_infer.py` 中已包含正确的自动重映射逻辑。
2. **显存占用**: Multi-Scale TTA 会占用较多显存，如遇 OOM 请在 `kfold_infer.py` 中调小 Batch Size。
