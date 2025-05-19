import pandas as pd
import numpy as np
import re
import os
import json
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
tqdm.pandas()

class CRAFT:
    SEED = 1
    
    # Train, Valid, Test 갯수에 대한 Configuration
    TRAIN_SIZE_PER_LEVEL = 300  # 주제 하나당 초등 300개, 중등 300개, 고등 300개 (총 갯수 300 * 3 * 5 = 4500)
    VALID_SIZE_PER_LEVEL  = 30   # 주제 하나당 초등 30개, 중등 30개, 고등 30개 (총 갯수 30 * 3 * 5 = 450)
    TEST_SIZE_PER_LEVEL  = 30    # 주제 하나당 초등 30개, 중등 30개, 고등 30개 (총 갯수 30 * 3 * 5 = 450)

    # 원시 데이터셋 경로
    DATA_PATH = './dataset/1.Training/label'

    # Total 데이터셋 전처리 이전, 이후 csv파일 경로
    DF_BEFORE_PREPROCESS_CSV_PATH = './df_total_before_preprocess.csv'
    DF_AFTER_PREPROCESS_CSV_PATH = './df_total_after_preprocess.csv'
    
    # 잔처리 이후 Human Dataset Train/Valid/Test csv파일 경로
    DF_HUMAN_TRAIN_CSV_PATH = './df_human_train.csv'
    DF_HUMAN_VALID_CSV_PATH = './df_human_valid.csv'
    DF_HUMAN_TEST_CSV_PATH = './df_human_test.csv'
    
    # 주제
    TRAIN_TOPICS = ['TL_글짓기/글짓기', 'TL_대안제시/대안제시', 'TL_설명글/설명글', 'TL_주장/주장', 'TL_찬성반대/찬성반대']

combined_data = []
for topic in tqdm(CRAFT.TRAIN_TOPICS):
    topic_path = os.path.join(CRAFT.DATA_PATH, topic)
    files = [file for file in os.listdir(topic_path) if file.endswith('.json')]

    for file in files:
        file_path = os.path.join(topic_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            # 기본 정보
            essay_id = content['info']['essay_id']
            essay_prompt = content['info']['essay_prompt']
            essay_type = content['info']['essay_type']
            essay_main_subject = content['info']['essay_main_subject']
            
            # 초,중,고 & 학년
            student_grade = content['student']['student_grade']
            
            # paragraph_txt가 여러개로 나뉜 경우 통합
            paragraphs = content['paragraph']
            combined_paragraph_txt = ''.join(p['paragraph_txt'] for p in paragraphs)
            
            combined_data.append({
                "essay_id": essay_id,
                "student_grade": student_grade,
                "essay_type": essay_type,
                "essay_main_subject": essay_main_subject,
                "essay_prompt": essay_prompt,
                "paragraph_txt": combined_paragraph_txt
            })

combined_df = pd.DataFrame(combined_data)


# preprocess 전 통합 데이터프레임 저장
combined_df.to_csv(CRAFT.DF_BEFORE_PREPROCESS_CSV_PATH, index=False, encoding='utf-8-sig')
combined_df.shape

if combined_df.isnull().values.any():
    print("NaN 존재함")
    combined_df = combined_df.dropna()
else:
    print("NaN 존재하지 않음")

combined_df['paragraph_length'] = combined_df['paragraph_txt'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(combined_df['paragraph_length'], bins=150, edgecolor='black', alpha=0.7)
plt.title("Distribution of Paragraph Lengths", fontsize=16)
plt.xlabel("Paragraph Length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# paragraph의 길이가 1000 초과인 경우 제거
combined_df = combined_df[combined_df['paragraph_length'] <= 1000]

# paragraph의 길이가 300 미만인 경우 제거
combined_df = combined_df[combined_df['paragraph_length'] >= 300]

combined_df.shape

def count_patterns(text):
    ''' #로 시작하고 #로 끝나는 태그 추출 '''
    matches = re.findall(r'#.*?#', text)
    pattern_counter.update(matches)

pattern_counter = Counter()
combined_df['paragraph_txt'].apply(count_patterns)

# 많은 순으로 정렬
sorted_patterns = pattern_counter.most_common()

for pattern, count in sorted_patterns:
    print(f"{pattern}: {count}번 등장")

def clean_text(text, remove_emoji=True, remove_url=True):
    # HTML 태그 제거
    text = BeautifulSoup(text, "html.parser").get_text()
    # 중복 공백/줄바꿈 정리
    text = re.sub(r'\s+', ' ', text).strip()
    # 이모지 제거
    if remove_emoji:
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # URL 제거
    if remove_url:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

# 데이터 전처리 함수화

def preprocess_paragraphs(df, remove_emoji=True, remove_url=True):
    df = df.copy()
    df['paragraph_txt'] = df['paragraph_txt'].astype(str).apply(lambda x: clean_text(x, remove_emoji, remove_url))
    return df

# 전처리 적용
combined_df = preprocess_paragraphs(combined_df)

# preprocess 후 통합 데이터프레임 저장
combined_df.to_csv(CRAFT.DF_AFTER_PREPROCESS_CSV_PATH, index=False, encoding='utf-8-sig')