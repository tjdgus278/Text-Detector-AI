import pandas as pd
import numpy as np
import re
import os
import json
import matplotlib.pyplot as plt
import random

from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm

tqdm.pandas()

class CRAFT:
    SEED = 1
    
    TRAIN_SIZE_PER_LEVEL = 200
    VALID_SIZE_PER_LEVEL  = 20
    TEST_SIZE_PER_LEVEL  = 20

    DATA_PATH = './dataset/label'

    DF_BEFORE_PREPROCESS_CSV_PATH = './df_total_before_preprocess.csv'
    DF_AFTER_PREPROCESS_CSV_PATH = './df_total_after_preprocess.csv'
    
    DF_HUMAN_TRAIN_CSV_PATH = './df_human_train.csv'
    DF_HUMAN_VALID_CSV_PATH = './df_human_valid.csv'
    DF_HUMAN_TEST_CSV_PATH = './df_human_test.csv'
    
    TRAIN_TOPICS = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']

os.makedirs(os.path.dirname(CRAFT.DF_BEFORE_PREPROCESS_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CRAFT.DF_AFTER_PREPROCESS_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CRAFT.DF_HUMAN_TRAIN_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CRAFT.DF_HUMAN_VALID_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CRAFT.DF_HUMAN_TEST_CSV_PATH), exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
set_seed(CRAFT.SEED)

def clean_text(text, remove_emoji=True, remove_url=True):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'#.*?#', '', text)
    text = re.sub(r'[“”‘’"\'`]', '', text)
    if remove_emoji:
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    if remove_url:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_essay_row(json_data):
    info = json_data.get("info", {})
    rubric = json_data.get("rubric", {})
    grade = info.get("student_grade") or rubric.get("essay_grade")
    essay_type = info.get("essay_type") or rubric.get("essay_type")
    subject = info.get("essay_main_subject") or rubric.get("essay_main_subject")
    prompt = info.get("essay_prompt")
    prompt = clean_text(prompt) if prompt else ''
    paragraphs = json_data.get("paragraph", [])
    text = " ".join([clean_text(p.get("paragraph_txt", "")) for p in paragraphs])
    return {
        "grade": grade,
        "type": essay_type,
        "subject": subject,
        "prompt": prompt,
        "text": text,
        "label": 1
    }

combined_data = []
for topic in os.listdir(CRAFT.DATA_PATH):
    topic_path = os.path.join(CRAFT.DATA_PATH, topic)
    if not os.path.isdir(topic_path):
        continue
    for file in os.listdir(topic_path):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(topic_path, file), 'r', encoding='utf-8') as f:
            content = json.load(f)
            row = extract_essay_row(content)
            combined_data.append(row)

combined_df = pd.DataFrame(combined_data)
# ====== 아래 코드 삭제 ======
# combined_df.to_csv(CRAFT.DF_BEFORE_PREPROCESS_CSV_PATH, index=False, encoding='utf-8-sig')
# print(f"Saved raw data to {CRAFT.DF_BEFORE_PREPROCESS_CSV_PATH}, shape: {combined_df.shape}")

if combined_df.isnull().values.any():
    print("NaN 존재함 -> 제거")
    combined_df = combined_df.dropna()
else:
    print("NaN 존재하지 않음")

combined_df['paragraph_length'] = combined_df['text'].apply(len)
plt.figure(figsize=(10, 6))
plt.hist(combined_df['paragraph_length'], bins=150, edgecolor='black', alpha=0.7)
plt.title("Distribution of Paragraph Lengths", fontsize=16)
plt.xlabel("Paragraph Length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

combined_df = combined_df[(combined_df['paragraph_length'] <= 1000) & (combined_df['paragraph_length'] >= 300)]
combined_df = combined_df.drop(columns=['paragraph_length'])

combined_df["text"] = combined_df["text"].astype(str).apply(lambda x: clean_text(x))
combined_df["numbering"] = range(1, len(combined_df) + 1)
combined_df = combined_df[["grade", "type", "subject", "prompt", "text", "label"]]

# grade 통일
def map_grade(grade):
    if '초등' in grade:
        return '초등'
    elif '중학교' in grade or '중등' in grade:
        return '중등'
    elif '고등' in grade:
        return '고등'
    else:
        return '기타'
combined_df['grade'] = combined_df['grade'].apply(map_grade)

# 기타는 제거
combined_df = combined_df[combined_df['grade'] != '기타']

# ⬇️ 로그 출력 시작 ⬇️
print("\n========== grade 값 분포 ==========")
print(combined_df['grade'].value_counts())
print("\n========== subject 값 수 (유니크) ==========")
print(len(combined_df['subject'].unique()))
print("========== subject 유니크 리스트 ==========")
print(sorted(combined_df['subject'].unique()))
print("\n========== (grade, subject) 그룹 수 ==========")
print(combined_df.groupby(['grade', 'subject']).ngroups)
print("====================================\n")

# ====== 아래 코드 삭제 ======
# combined_df.to_csv(CRAFT.DF_AFTER_PREPROCESS_CSV_PATH, index=False, encoding='utf-8-sig')
# print(f"Saved cleaned data to {CRAFT.DF_AFTER_PREPROCESS_CSV_PATH}, shape: {combined_df.shape}")

def stratified_group_sample(df, train_n, valid_n, test_n, group_cols=['grade', 'subject']):
    train_list, valid_list, test_list = [], [], []
    grouped = df.groupby(group_cols)
    for _, group in grouped:
        group = group.sample(frac=1, random_state=CRAFT.SEED)
        n = len(group)
        n_train = min(train_n, n)
        n_valid = min(valid_n, n - n_train)
        n_test = min(test_n, n - n_train - n_valid)
        train = group.iloc[:n_train]
        valid = group.iloc[n_train:n_train + n_valid]
        test = group.iloc[n_train + n_valid:n_train + n_valid + n_test]
        train_list.append(train)
        valid_list.append(valid)
        test_list.append(test)
    return pd.concat(train_list).reset_index(drop=True), pd.concat(valid_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)

# Stratified split 기준을 ['grade', 'type']으로 변경
train_df, valid_df, test_df = stratified_group_sample(
    combined_df, 
    CRAFT.TRAIN_SIZE_PER_LEVEL, 
    CRAFT.VALID_SIZE_PER_LEVEL, 
    CRAFT.TEST_SIZE_PER_LEVEL,
    group_cols=['grade', 'type']
)

train_df = train_df.drop_duplicates()
valid_df = valid_df.drop_duplicates()
test_df = test_df.drop_duplicates()

def log_distribution(df, name):
    dist = df.groupby(['grade', 'type']).size()
    print(f"{name} 분포:")
    print(dist)
    with open(f"{name}_distribution.log", "w", encoding="utf-8") as f:
        f.write(str(dist))

log_distribution(train_df, "train")
log_distribution(valid_df, "valid")
log_distribution(test_df, "test")

train_df.to_csv(CRAFT.DF_HUMAN_TRAIN_CSV_PATH, index=False, encoding='utf-8-sig')
valid_df.to_csv(CRAFT.DF_HUMAN_VALID_CSV_PATH, index=False, encoding='utf-8-sig')
test_df.to_csv(CRAFT.DF_HUMAN_TEST_CSV_PATH, index=False, encoding='utf-8-sig')
print("Train/Valid/Test CSV 저장 완료")
print(f"train_df: {train_df.shape}, valid_df: {valid_df.shape}, test_df: {test_df.shape}")
