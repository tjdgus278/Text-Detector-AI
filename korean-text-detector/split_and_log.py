import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 파일 경로
DATA_PATH = './df_total_after_preprocess.csv'
TRAIN_PATH = './df_human_train.csv'
VALID_PATH = './df_human_valid.csv'
TEST_PATH = './df_human_test.csv'

SEED = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

# 데이터 로드
assert os.path.exists(DATA_PATH), f"{DATA_PATH} 파일이 존재하지 않습니다."
df = pd.read_csv(DATA_PATH)

# 셔플
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Stratified Split: 에세이 타입+학년 기준
stratify_col = df['글양식'] + '_' + df['학년']
train_valid_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=stratify_col)
stratify_col_tv = train_valid_df['글양식'] + '_' + train_valid_df['학년']
train_df, valid_df = train_test_split(train_valid_df, test_size=VALID_SIZE/(1-TEST_SIZE), random_state=SEED, stratify=stratify_col_tv)

# 저장
train_df.to_csv(TRAIN_PATH, index=False, encoding='utf-8-sig')
valid_df.to_csv(VALID_PATH, index=False, encoding='utf-8-sig')
test_df.to_csv(TEST_PATH, index=False, encoding='utf-8-sig')

# 분배 현황 로그
for split_name, split_df in zip(['Train', 'Valid', 'Test'], [train_df, valid_df, test_df]):
    logging.info(f'[{split_name}] 전체 샘플 수: {len(split_df)}')
    logging.info(f'[{split_name}] 에세이 타입/학년별 분포:')
    logging.info('\n' + str(split_df.groupby(['글양식', '학년']).size()))
