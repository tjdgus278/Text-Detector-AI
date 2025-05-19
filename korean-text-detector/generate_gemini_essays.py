import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
import time

# .env에서 API 키 불러오기
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
assert GOOGLE_API_KEY, "GOOGLE_API_KEY가 .env에 없습니다."
genai.configure(api_key=GOOGLE_API_KEY)

HUMAN_PATH = './df_total_after_preprocess.csv'
AI_PATH = './df_ai_after_preprocess.csv'

df = pd.read_csv(HUMAN_PATH)
df['label'] = 0

def generate_gemini_essay(prompt, max_retry=3, sleep_sec=2):
    for attempt in range(max_retry):
        try:
            # 최신 모델명 확인 및 사용 (예: 'models/gemini-pro')
            model = genai.GenerativeModel('models/gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API 오류: {e}. {attempt+1}/{max_retry}회 재시도...")
            time.sleep(sleep_sec)
    return ''

ai_texts = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['prompt']
    ai_essay = generate_gemini_essay(prompt)
    ai_texts.append(ai_essay)

df['text'] = ai_texts
df = df[['grade', 'type', 'subject', 'prompt', 'text', 'label']]
df.to_csv(AI_PATH, index=False, encoding='utf-8-sig')
print(f"AI 에세이 데이터셋 저장 완료: {AI_PATH}")

# 사용법:
# 1. .env 파일에 GOOGLE_API_KEY=... 형태로 저장
# 2. pip install google-generativeai python-dotenv tqdm
# 3. python generate_gemini_essays.py
