import pandas as pd
import numpy as np
import os
from collections import Counter

# 1. 加载训练集分布 (作为目标分布)
# 数据集是 ImageFolder 格式，直接统计文件夹内的图片数量
train_dir = "../fer_data/train"
class_map = {
    "Angry": 0,
    "Fear": 1,
    "Happy": 2,
    "Neutral": 3,
    "Sad": 4,
    "Surprise": 5
}

train_counts = Counter()
total_train = 0

if os.path.exists(train_dir):
    print("Scanning training directories...")
    for class_name, label_idx in class_map.items():
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            train_counts[label_idx] = count
            total_train += count
else:
    print(f"Error: Training directory {train_dir} not found.")
    exit(1)

target_dist = {k: v / total_train for k, v in train_counts.items()}

print("Target Distribution (Train):")
for k in sorted(target_dist.keys()):
    # 反向查找类名
    class_name = [name for name, idx in class_map.items() if idx == k][0]
    print(f"Class {k} ({class_name}): {target_dist[k]:.4f}")

# 2. 加载预测结果
sub_df = pd.read_csv("../submission.csv")
# 这里我们没有原始概率，只有最终预测结果。
# 如果要进行严格的分布对齐，通常需要修改 kfold_infer.py 保存概率文件。
# 但我们可以做一个简单的统计检查。

pred_counts = Counter(sub_df['Emotion'])
total_pred = len(sub_df)
pred_dist = {k: v / total_pred for k, v in pred_counts.items()}

print("\nPrediction Distribution (Test):")
for k in sorted(pred_dist.keys()):
    print(f"Class {k}: {pred_dist.get(k, 0):.4f}")

print("\nDistribution Difference (Pred - Target):")
for k in sorted(target_dist.keys()):
    diff = pred_dist.get(k, 0) - target_dist[k]
    print(f"Class {k}: {diff:+.4f}")

print("\n提示: 如果某些类别的差异特别大 (例如 > 0.05)，说明模型可能存在系统性偏差。")
print("由于我们目前只保存了最终类别而没有保存概率矩阵，无法进行后处理对齐。")
print("如果需要进一步提升，建议修改 kfold_infer.py 保存 npy 格式的概率矩阵。")
