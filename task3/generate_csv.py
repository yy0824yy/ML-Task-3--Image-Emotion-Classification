import os
import pandas as pd

train_dir = "../fer_data/train"
test_dir = "../fer_data/test"
train_csv = "../fer_data/train.csv"
submission_csv = "../fer_data/sample_submission.csv"

# Define class mapping
# Sorted alphabetically: Angry, Fear, Happy, Neutral, Sad, Surprise
classes = sorted(os.listdir(train_dir))
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
print("Class mapping:", class_to_idx)

# Generate train.csv
train_data = []
for cls_name in classes:
    cls_dir = os.path.join(train_dir, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Path relative to train_dir
            rel_path = os.path.join(cls_name, img_name)
            train_data.append([rel_path, class_to_idx[cls_name]])

df_train = pd.DataFrame(train_data, columns=['filename', 'label'])
df_train.to_csv(train_csv, index=False)
print(f"Generated {train_csv} with {len(df_train)} images.")

# Generate sample_submission.csv
test_data = []
if os.path.exists(test_dir):
    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_data.append([img_name, 0]) # Dummy label

df_test = pd.DataFrame(test_data, columns=['filename', 'label'])
df_test.to_csv(submission_csv, index=False)
print(f"Generated {submission_csv} with {len(df_test)} images.")
