import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from our previous analysis
labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
train_dist = [0.1402, 0.1449, 0.2544, 0.1754, 0.1720, 0.1132]
pred_dist = [0.1229, 0.1222, 0.2582, 0.1904, 0.1182, 0.1883]

x = np.arange(len(labels))
width = 0.35

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, train_dist, width, label='Train Distribution (Ground Truth)', color='#7fcdbb')
rects2 = ax.bar(x + width/2, pred_dist, width, label='Test Prediction (Model Output)', color='#2c7fb8')

# Decorate
ax.set_ylabel('Percentage (%)')
ax.set_title('Class Distribution Mismatches: Train vs Test Predictions', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()
ax.set_ylim(0, 0.30)

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('task3/class_distribution_comparison.png', dpi=300)
print("Chart generated: task3/class_distribution_comparison.png")
