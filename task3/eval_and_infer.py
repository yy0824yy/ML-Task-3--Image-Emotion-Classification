import os
from datetime import datetime

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import Config
from dataset import get_test_loader, get_data_loaders
from model import get_emotion_model


def _get_class_names_from_loader(loader):
    # Try to get class names from nested datasets (Subset(ImageFolder))
    ds = getattr(loader, 'dataset', None)
    classes = None
    if ds is not None:
        # Subset
        base = getattr(ds, 'dataset', None)
        if base is not None and hasattr(base, 'classes'):
            classes = base.classes
        elif hasattr(ds, 'classes'):
            classes = ds.classes
    if classes is None:
        classes = [str(i) for i in range(Config.NUM_CLASSES)]
    return classes


def run_inference(model):
    test_loader = get_test_loader()
    model.eval()
    preds, fnames = [], []
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Infer(Test)'):
            images = images.to(Config.DEVICE)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.detach().cpu().tolist())
            fnames.extend(names)

    # 输出符合示例的列名
    submission = pd.DataFrame({'ID': [str(x) for x in fnames], 'Emotion': [int(x) for x in preds]})
    submission.to_csv(Config.RESULT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    return Config.RESULT_CSV


def evaluate_on_val(model):
    _, val_loader = get_data_loaders()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Eval(Val)'):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    cls_names = _get_class_names_from_loader(val_loader)
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in cls_names], digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return acc, report, cm


def append_to_log(val_acc, report, cm, submission_path):
    log_path = os.path.join(os.path.dirname(__file__), 'model_optimization_log.txt')
    lines = []
    lines.append('\n')
    lines.append(f"评估与推理记录（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）\n")
    lines.append(f"使用权重：{Config.MODEL_SAVE_PATH}\n")
    lines.append(f"验证集准确率：{val_acc*100:.2f}%\n")
    lines.append("分类报告：\n")
    lines.append(report + "\n")
    lines.append("混淆矩阵：\n")
    # 格式化混淆矩阵为紧凑行
    for row in cm:
        lines.append(' '.join(map(str, row)) + "\n")
    lines.append(f"提交文件：{submission_path}\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    print(f"Using Device: {Config.DEVICE}")
    # Load model and weights
    model = get_emotion_model()
    print(f"Loading weights from {Config.MODEL_SAVE_PATH}")
    state = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)

    # 1) Inference on test set -> submission.csv
    sub_path = run_inference(model)
    print(f"Submission saved to: {sub_path}")

    # 2) Evaluate on validation set
    val_acc, report, cm = evaluate_on_val(model)
    print(f"Val ACC: {val_acc*100:.2f}%")

    # 3) Append results to log
    append_to_log(val_acc, report, cm, sub_path)
    print("Evaluation and log update complete.")


if __name__ == '__main__':
    main()
