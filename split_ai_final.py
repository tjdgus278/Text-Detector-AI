import pandas as pd

def clean_grade(val):
    # '고등_1학년' -> '고등', '중등_2학년' -> '중등', etc.
    if pd.isnull(val):
        return ''
    return str(val).split('_')[0]

def convert_and_save(src, dst):
    df = pd.read_csv(src)
    label_col = 'label'
    # 컬럼명 자동 매핑
    grade_col = 'grade' if 'grade' in df.columns else 'student_grade'
    type_col = 'type' if 'type' in df.columns else 'essay_type'
    subject_col = 'subject' if 'subject' in df.columns else 'essay_main_subject'
    prompt_col = 'prompt' if 'prompt' in df.columns else 'essay_prompt'
    text_col = 'text'
    # label==1만 필터링
    df_ai = df[df[label_col] == 1].copy()
    # grade에서 _n학년 등 제거
    df_ai['grade'] = df_ai[grade_col].apply(clean_grade)
    # 컬럼명 변환 및 순서 맞추기
    df_ai = df_ai.rename(columns={
        type_col: 'type',
        subject_col: 'subject',
        prompt_col: 'prompt',
        text_col: 'text',
    })
    df_ai['label'] = 0
    df_ai = df_ai[['grade', 'type', 'subject', 'prompt', 'text', 'label']]
    df_ai.to_csv(dst, index=False, encoding='utf-8-sig')
    print(f"{src} → {dst}: {len(df_ai)}개 저장 (grade 정제, label=0)")

for src, dst in [
    ('df_final_train_v1.csv', 'df_ai_train.csv'),
    ('df_final_valid_v1.csv', 'df_ai_valid.csv'),
    ('df_final_test_v1.csv', 'df_ai_test.csv')
]:
    convert_and_save(src, dst)
