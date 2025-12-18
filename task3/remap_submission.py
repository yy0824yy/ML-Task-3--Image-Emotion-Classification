import pandas as pd
import os
from config import Config

def remap():
    # 读取我们刚刚生成的提交文件
    # 假设当前 submission.csv 是基于字母序 (Angry, Fear, Happy, Neutral, Sad, Surprise) 预测的
    input_csv = Config.RESULT_CSV
    output_csv = "submission_remapped.csv"
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found!")
        return

    df = pd.read_csv(input_csv)
    
    # 映射逻辑：
    # 我们的模型 (字母序): 
    # 0: Angry
    # 1: Fear
    # 2: Happy
    # 3: Neutral
    # 4: Sad
    # 5: Surprise
    
    # 推测的 Kaggle 目标 (FER2013 经典顺序，去除 Disgust 后):
    # 0: Angry
    # 1: Fear   (原 2)
    # 2: Happy  (原 3)
    # 3: Sad    (原 4)
    # 4: Surprise (原 5)
    # 5: Neutral (原 6)
    
    # 映射字典: {原值 : 新值}
    mapping = {
        0: 0, # Angry -> Angry
        1: 1, # Fear -> Fear
        2: 2, # Happy -> Happy
        3: 5, # Neutral -> Neutral (变动!)
        4: 3, # Sad -> Sad (变动!)
        5: 4  # Surprise -> Surprise (变动!)
    }
    
    # 应用映射
    # 注意：必须确保 Emotion 列是 int 类型
    df['Emotion'] = df['Emotion'].map(mapping)
    
    # 保存
    df.to_csv(output_csv, index=False, encoding='utf-8', lineterminator='\n')
    print(f"Remapped submission saved to {output_csv}")
    print("Mapping applied: 3->5 (Neutral), 4->3 (Sad), 5->4 (Surprise)")

if __name__ == "__main__":
    remap()
