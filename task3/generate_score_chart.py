import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Data
stages = ['Baseline\n(ResNet)', 'Ensemble\n(Heterogeneous)', 'Pseudo-Labeling\n(+Weighted)', 'Final\n(Multi-Scale TTA)']
scores = [0.731, 0.739, 0.744, 0.745]

# Create bar plot
bars = plt.bar(stages, scores, color=['#a8dab5', '#7fcdbb', '#41b6c4', '#225ea8'], width=0.5)

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize
plt.ylim(0.70, 0.755)
plt.ylabel('Kaggle Private Score', fontsize=12)
plt.title('Performance Improvements across Optimization Stages', fontsize=14, pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save
plt.tight_layout()
plt.savefig('task3/dataset_score_trend.png', dpi=300)
print("Chart generated: task3/dataset_score_trend.png")
