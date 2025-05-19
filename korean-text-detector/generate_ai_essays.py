import os
import openai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time

# .env 파일에서 OPENAI_API_KEY 불러오기 (보안)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
assert OPENAI_API_KEY, 'OPENAI_API_KEY가 환경변수 또는 .env에 설정되어 있어야 합니다.'
openai.api_key = OPENAI_API_KEY

# 기존 사람 데이터셋 경로 (영문 컬럼)
HUMAN_PATH = './df_total_after_preprocess.csv'
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

# tqdm 진행바로 GPT 에세이 생성 (prompt 컬럼 활용)
ai_texts = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['prompt']
    ai_essay = generate_gpt_essay(prompt)
    ai_texts.append(ai_essay)

df['text'] = ai_texts

# 결과 저장 (기존 구조와 동일)
df = df[['grade', 'type', 'subject', 'prompt', 'text', 'label']]
df.to_csv(AI_PATH, index=False, encoding='utf-8-sig')
print(f"AI 에세이 데이터셋 저장 완료: {AI_PATH}")

# 사용법:
# 1. .env 파일에 OPENAI_API_KEY=sk-... 형태로 저장
# 2. pip install openai python-dotenv tqdm
# 3. python generate_ai_essays.py
