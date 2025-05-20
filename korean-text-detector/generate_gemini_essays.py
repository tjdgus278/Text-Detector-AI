import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

# .env에서 API 키 불러오기
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
assert GOOGLE_API_KEY, "GOOGLE_API_KEY가 .env에 없습니다."
genai.configure(api_key=GOOGLE_API_KEY)
print(list(genai.list_models()))

HUMAN_PATH = './df_human_train.csv'
AI_PATH = './df_ai_after_preprocess.csv'

df = pd.read_csv(HUMAN_PATH)
df['label'] = 0

def generate_gemini_essay(prompt, school, max_retry=3, sleep_sec=2):
    # '중등'이면 '중학교', '초등'이면 '초등학교', '고등'이면 '고등학교'로 변환
    if school == '중등':
        school_name = '중학교'
    elif school == '초등':
        school_name = '초등학교'
    elif school == '고등':
        school_name = '고등학교'
    else:
        school_name = school  # 예외적으로 다른 값이 들어오면 그대로 사용
    for attempt in range(max_retry):
        try:
            # 학교 수준에 따라 다르게 설정하는 시스템 프롬프트
            grade_system_prompt = f"너는 대한민국의 {school_name} 학생이다. 마크다운 용법을 사용하지 말고 학생이 글을 쓰듯이 답하시오. 아래 프롬프트에 따라 자연스럽고 학생다운 한국어 에세이를 써줘. 문장 길이, 어휘 수준, 감정 표현, 문체 등도 실제 학생처럼 써야 해."
            model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp-1219')
            response = model.generate_content(f"{grade_system_prompt}\n\n프롬프트: {prompt}")
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API 오류: {e}. {attempt+1}/{max_retry}회 재시도...")
            time.sleep(sleep_sec)
    return ''

# (1) grade/type별 600/60/60 샘플링용 리스트 생성

grade_list = ['초등', '중등', '고등']
type_list = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']
SPLIT_COUNTS = {'train': 40, 'valid': 4, 'test': 4}

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
    ai_texts = [None] * len(split_df)

    def task(row):
        return generate_gemini_essay(row['prompt'], row['grade'])

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(task, row): i for i, row in split_df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                ai_texts[i] = future.result()
            except Exception as e:
                ai_texts[i] = ''
                print(f"Error at {i}: {e}")

    split_df['text'] = ai_texts
    split_df['source'] = 'gemini'  # source 컬럼 추가
    split_df = split_df[['grade', 'type', 'subject', 'prompt', 'text', 'label', 'source']]
    split_df.to_csv(f'./df_ai_gemini_{split}.csv', index=False, encoding='utf-8-sig')
    print(f"AI {split} 데이터셋 저장 완료: ./df_ai_gemini_{split}.csv")
print(list(genai.list_models()))

# 사용법:
# 1. .env 파일에 GOOGLE_API_KEY=... 형태로 저장
# 2. pip install google-generativeai python-dotenv tqdm
# 3. python generate_gemini_essays.py
