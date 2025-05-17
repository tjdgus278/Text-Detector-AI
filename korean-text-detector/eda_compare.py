import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import re

# 텍스트 길이 분포 & 어휘 다양도(타입/토큰 비율) 비교 함수

def plot_text_length_distribution(df1, df2, col, label1="Dataset 1", label2="Dataset 2"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df1[col].str.len(), bins=50, color='blue', label=label1, kde=True, stat="density", alpha=0.5)
    sns.histplot(df2[col].str.len(), bins=50, color='orange', label=label2, kde=True, stat="density", alpha=0.5)
    plt.title(f"Text Length Distribution: {label1} vs {label2}")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def vocab_richness(df, col):
    all_tokens = []
    for text in df[col]:
        tokens = re.findall(r'\w+', str(text).lower())
        all_tokens.extend(tokens)
    if not all_tokens:
        return 0
    unique_tokens = set(all_tokens)
    return len(unique_tokens) / len(all_tokens)


def compare_vocab_richness(df1, df2, col, label1="Dataset 1", label2="Dataset 2"):
    richness1 = vocab_richness(df1, col)
    richness2 = vocab_richness(df2, col)
    print(f"Vocab Richness (Type/Token Ratio) - {label1}: {richness1:.4f}")
    print(f"Vocab Richness (Type/Token Ratio) - {label2}: {richness2:.4f}")


def get_top_n_words(df, col, n=20):
    all_tokens = []
    for text in df[col]:
        tokens = re.findall(r'\w+', str(text).lower())
        all_tokens.extend(tokens)
    counter = Counter(all_tokens)
    return counter.most_common(n)


def compare_top_words(df1, df2, col, n=20, label1="Dataset 1", label2="Dataset 2"):
    top1 = get_top_n_words(df1, col, n)
    top2 = get_top_n_words(df2, col, n)
    print(f"Top {n} words in {label1}:")
    for word, count in top1:
        print(f"{word}: {count}")
    print("\n" + "-"*40 + "\n")
    print(f"Top {n} words in {label2}:")
    for word, count in top2:
        print(f"{word}: {count}")

# 사용 예시 (아래 부분을 실제 데이터프레임에 맞게 수정해서 사용)
if __name__ == "__main__":
    # 예시: 두 개의 전처리된 csv 파일을 불러와 비교
    df1 = pd.read_csv("./df_human_train.csv")
    df2 = pd.read_csv("./df_ai_train.csv")
    
    # 텍스트 길이 분포 비교
    plot_text_length_distribution(df1, df2, col="paragraph_txt", label1="Human", label2="AI")
    
    # 어휘 다양도 비교
    compare_vocab_richness(df1, df2, col="paragraph_txt", label1="Human", label2="AI")
    
    # 고빈도 단어 비교
    print("\n[Top 20 Frequent Words]")
    compare_top_words(df1, df2, col="paragraph_txt", n=20, label1="Human", label2="AI")
