import pandas as pd

def stratified_sample(src, dst, n_per_group, grade_col='grade', type_col='type'):
    df = pd.read_csv(src)
    out = []
    grade_list = ['초등', '중등', '고등']
    type_list = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']
    for grade in grade_list:
        for etype in type_list:
            sub = df[(df[grade_col] == grade) & (df[type_col] == etype)]
            sub = sub.sample(n=min(n_per_group, len(sub)), random_state=42)
            out.append(sub)
    df_out = pd.concat(out, ignore_index=True)
    df_out.to_csv(dst, index=False, encoding='utf-8-sig')
    print(f"{dst}: {len(df_out)}개 저장 (조합별 최대 {n_per_group}개)")

stratified_sample('df_ai_train.csv', 'df_ai_train_3000.csv', 200)
stratified_sample('df_ai_valid.csv', 'df_ai_valid_300.csv', 20)
stratified_sample('df_ai_test.csv', 'df_ai_test_300.csv', 20)
