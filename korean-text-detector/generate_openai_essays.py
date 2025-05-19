import os
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
from itertools import product

# .env 파일에서 OPENAI_API_KEY 불러오기 (보안)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
assert OPENAI_API_KEY, 'OPENAI_API_KEY가 환경변수 또는 .env에 설정되어 있어야 합니다.'
openai.api_key = OPENAI_API_KEY

# 기존 사람 데이터셋 경로 (영문 컬럼)
HUMAN_PATH = '../df_human_train.csv'  # 또는 적절한 human 데이터셋 경로로 변경
AI_PATH = './df_ai_after_preprocess.csv'

# 데이터 로드 (영문 컬럼)
df = pd.read_csv(HUMAN_PATH)

# label=0으로 변경
df['label'] = 0

# GPT로 에세이 생성 함수
def generate_gpt_essay(prompt, max_retry=3, sleep_sec=2):
    for attempt in range(max_retry):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Korean student. Write a Korean essay based on the following prompt. The style should be natural and similar to a student's writing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API 오류: {e}. {attempt+1}/{max_retry}회 재시도...")
            time.sleep(sleep_sec)
    return ''

# (1) grade/type별 600/60/60 샘플링용 리스트 생성
grade_list = ['초등', '중등', '고등']
type_list = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']
SPLIT_COUNTS = {'train': 600, 'valid': 60, 'test': 60}

# (2) 샘플링 인덱스 수집
split_indices = {'train': [], 'valid': [], 'test': []}
for grade, etype in product(grade_list, type_list):
    sub_df = df[(df['grade'] == grade) & (df['type'] == etype)]
    sub_df = sub_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    n = len(sub_df)
    n_train = min(SPLIT_COUNTS['train'], n)
    n_valid = min(SPLIT_COUNTS['valid'], n - n_train)
    n_test = min(SPLIT_COUNTS['test'], n - n_train - n_valid)
    split_indices['train'].extend(sub_df.index[:n_train])
    split_indices['valid'].extend(sub_df.index[n_train:n_train+n_valid])
    split_indices['test'].extend(sub_df.index[n_train+n_valid:n_train+n_valid+n_test])

# (3) split별 DataFrame 생성
for split in ['train', 'valid', 'test']:
    split_df = df.loc[split_indices[split]].copy().reset_index(drop=True)
    ai_texts = []
    for i, row in tqdm(split_df.iterrows(), total=len(split_df)):
        prompt = row['prompt']
        ai_essay = generate_gpt_essay(prompt)
        print(f"[{i+1}/{len(split_df)}] prompt: {prompt}\nAI essay: {ai_essay}\n{'-'*40}")
        ai_texts.append(ai_essay)
    split_df['text'] = ai_texts
    split_df['source'] = 'openai'  # source 컬럼 추가
    split_df = split_df[['grade', 'type', 'subject', 'prompt', 'text', 'label', 'source']]
    split_df.to_csv(f'./df_ai_openai_{split}.csv', index=False, encoding='utf-8-sig')
    print(f"AI {split} 데이터셋 저장 완료: ./df_ai_openai_{split}.csv")

# 사용법:
# 1. .env 파일에 OPENAI_API_KEY=sk-... 형태로 저장
# 2. pip install openai python-dotenv tqdm
# 3. python generate_ai_essays.py
