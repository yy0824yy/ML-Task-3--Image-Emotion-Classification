# Task3 FER Training

## 项目简介
本项目用于人脸情绪识别（FER）的训练与推理，数据位于 `fer_data/`，训练代码位于 `task3/`。

## 文件结构
- `fer_data/`
  - `train/`：按类别子文件夹组织的训练图片（Angry, Fear, Happy, Neutral, Sad, Surprise）
  - `test/`：测试图片（可选）
  - `train.csv`、`sample_submission.csv`：由脚本生成（可选）
- `task3/`
  - `config.py`：超参数与路径配置
  - `dataset.py`：数据加载与增强（训练/验证分离的 transforms，训练带均衡采样）
  - `model.py`：模型定义（ResNet50 + 加强分类头）
  - `train.py`：训练脚本（AdamW + Label Smoothing + OneCycleLR）
  - `inference.py`：推理脚本（可选）
  - `generate_csv.py`：从文件夹结构生成 CSV（可选）
  - `best_model.pth`：保存的最佳模型权重
  - `model_optimization_log.txt`：模型优化过程记录（每次改动都在此文件追加）

## 环境要求
- Python 3.10+
- 必需依赖：`torch`、`torchvision`、`tqdm`、`scikit-learn`、`pillow`、`numpy`、`pandas`、`matplotlib`（可选）
- GPU（建议）与 CUDA/cuDNN（如可用）

## 数据准备
- 将训练数据整理为：`fer_data/train/<ClassName>/*.jpg`（7 类）
- 测试数据（如需要）：`fer_data/test/*.jpg`
- 可使用脚本生成 CSV（非必须）：
```bash
cd /home/algo/chunzhuang/assign_gy/Task3/task3
python3 generate_csv.py
```

## 训练
```bash
cd /home/algo/chunzhuang/assign_gy/Task3/task3
python3 train.py
```
- 训练过程会在 `task3/best_model.pth` 保存验证集最佳权重。
- 超参数在 `config.py` 中配置：`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`IMAGE_SIZE`、设备等。

## 关键设计
- 验证集评估使用不含随机的数据增强（`val_transforms`）以确保指标稳定。
- 训练集采用更强的数据增强（`RandomResizedCrop`、`RandomErasing` 等），并使用 `WeightedRandomSampler` 平衡类别。
- 模型升级至 `ResNet50` 预训练，分类头增强（线性层 + ReLU + Dropout）。
- 优化器：`AdamW`；损失：`CrossEntropyLoss(label_smoothing=0.1)`；调度器：`OneCycleLR`。

## 推理（可选）
如需在测试集上推理并生成提交文件，请参考 `inference.py`（若需要我可补充脚本）。

## 报告与改动记录
- 每次模型或训练流程优化，都会将改动内容、文件位置、动机与预期、结果摘要，记录到：
  - `task3/model_optimization_log.txt`
- 你也可以在训练完成后，将混淆矩阵、每类 F1 等评估结果追加到该日志文件中。

## 常见问题
- 显存不足：将 `Config.BATCH_SIZE` 降至 16 或启用 AMP（可按需在训练脚本中加入）。
- 指标不稳定：确保验证集使用 `val_transforms`，并避免随机增强；必要时固定 `Config.SEED`。
- 训练过慢：检查是否启用 GPU（`Config.DEVICE` 输出），或使用更轻模型（如 ResNet18）对比。

## 复现实验模板
- 记录内容：数据版本、代码版本（提交哈希或日期）、超参数、训练时长、最终指标（ACC、F1）、混淆矩阵。
- 建议将关键图表与指标附到 `model_optimization_log.txt` 或单独的报告文档中。
