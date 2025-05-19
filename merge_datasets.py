import pandas as pd
import glob

# 통합 대상 파일명 패턴
ai_patterns = [
    './df_ai_openai_*.csv',
    './df_ai_gemini_*.csv',
    # 추후 다른 LLM별 파일 패턴 추가 가능
]

splits = ['train', 'valid', 'test']

# 1. AI 데이터 통합 (split별)
for split in splits:
    ai_dfs = []
    for pattern in ai_patterns:
        files = glob.glob(pattern.replace('*', split))
        for f in files:
            df = pd.read_csv(f)
            ai_dfs.append(df)
    if ai_dfs:
        df_ai_total = pd.concat(ai_dfs, ignore_index=True)
        df_ai_total.to_csv(f'./df_ai_total_{split}.csv', index=False, encoding='utf-8-sig')
        print(f'AI 통합: df_ai_total_{split}.csv ({len(df_ai_total)})')
    else:
        print(f'AI 데이터 없음: {split}')

# 2. human 데이터에 source 컬럼 추가 및 저장
for split in splits:
    df_human = pd.read_csv(f'./df_human_{split}.csv')
    df_human['source'] = 'human'
    df_human.to_csv(f'./df_human_{split}_with_source.csv', index=False, encoding='utf-8-sig')

# 3. human+AI 통합 (split별)
for split in splits:
    df_human = pd.read_csv(f'./df_human_{split}_with_source.csv')
    df_ai = pd.read_csv(f'./df_ai_total_{split}.csv') if glob.glob(f'./df_ai_total_{split}.csv') else pd.DataFrame()
    df_total = pd.concat([df_human, df_ai], ignore_index=True)
    df_total.to_csv(f'./df_total_{split}.csv', index=False, encoding='utf-8-sig')
    print(f'최종 통합: df_total_{split}.csv ({len(df_total)})')
