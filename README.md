AI 생성 콘텐츠 진위 판별 시스템 개발
전남대학교 소중단 디지털 경진대회 참가 프로젝트

본 프로젝트는 Hugging Face의 Transformer 모델인 team-lucid/deberta-v3-base-korean을 파인튜닝하여, AI가 작성한 텍스트와 사람이 작성한 텍스트를 구분하는 이진 분류 문제를 해결하는 것을 목표로 합니다.

1. Introduction
본 프로젝트는 전남대학교 소중단 디지털 경진대회 참가를 위해 진행하였습니다.
Python, transformers, datasets, scikit-learn 등의 라이브러리를 활용하여 한국어 텍스트가 AI에 의해 작성되었는지, 혹은 인간에 의해 작성되었는지를 분류하는 모델을 개발하였습니다.

Hugging Face의 team-lucid/deberta-v3-base-korean 사전학습 모델을 기반으로 파인튜닝을 진행하였으며, 데이터 전처리, 모델 학습, 성능 평가, 배포까지의 전 과정을 수행하였습니다.

2. 문제 정의 및 데이터 분석
🔹 문제 정의
생성형 AI의 확산으로 인해 AI가 작성한 글과 사람이 작성한 글을 구별하는 기술의 중요성이 점차 부각되고 있습니다.
특히 교육 환경에서는 학습자의 자율적인 사고와 표현 능력을 평가하기 위해, AI 개입을 최소화할 수 있는 검증 메커니즘이 필요합니다.

따라서 본 프로젝트는 주어진 텍스트가 사람이 직접 작성했는지, AI가 생성했는지를 분류하는 **이진 분류 문제(Binary Classification)**로 정의하였습니다.

🔹 사용한 데이터셋
사람 작성 텍스트: 국내 초·중·고 학생들이 실제로 작성한 자기소개서, 독후감, 수필 등을 수집하였습니다. 학령별로 글의 길이, 어휘 수준, 문체에서 차이를 보입니다.

AI 생성 텍스트: 동일한 주제 또는 유사 맥락에서 생성형 AI(ChatGPT 등)를 활용해 생성한 텍스트를 수집하였으며, 사람이 작성한 글과 최대한 유사한 주제와 스타일로 조정하였습니다.

🔹 데이터 전처리 및 EDA 결과
전처리 과정:
특수문자 제거, 띄어쓰기 정리, 중복 샘플 제거 등의 전처리를 수행하였으며, 학습 효율 향상을 위해 최소 길이 100자 제한을 설정하였습니다.

탐색적 분석(EDA):

사람 텍스트는 평균 350자, AI 텍스트는 평균 400자 분량으로 나타났습니다.

사람 텍스트는 감탄사, 반복 표현, 감정적이고 자유로운 문체의 특성이 있으며,
AI 텍스트는 더 논리적이고 정제된 구조를 보입니다.

시각화 분석을 통해 각 클래스의 단어 빈도, 문장 구조 다양성 등의 차이를 확인하였습니다.

🔹 분석 방향 설정
문체적 특성과 감정 표현 등에서 사람과 AI 텍스트 간의 차이가 확인됨에 따라,
본 프로젝트는 표현 다양성, 비형식성, 문장 길이, 어휘 패턴 등 문맥적 특징을 학습할 수 있는 Transformer 계열 모델에 기반한 분류를 시도합니다.
특히 DeBERTa v3는 문맥 파악 능력이 뛰어나 이 문제 해결에 적합하다고 판단하였습니다.

3. 해결 아이디어 및 모델 구현
🔹 사용한 모델
모델명: team-lucid/deberta-v3-base-korean

기반 아키텍처: DeBERTa v3 (Decoding-enhanced BERT with disentangled attention)

모델 특징:

단어 위치와 내용을 분리하여 이해하는 Disentangled Attention 구조

Enhanced Mask Decoder를 활용해 마스킹 복원 성능을 강화

기존 BERT, RoBERTa 대비 정합성과 일관성이 우수함

🔹 모델 설계 이유와 구조
단순 키워드 기반이 아닌, 감정 표현, 문체 다양성, 창의성 등 고차원 언어 특성을 포착해야 하므로,
문맥 이해력이 뛰어난 Transformer 계열 모델을 채택하였습니다.

DeBERTa는 문맥 흐름과 표현의 자연스러움을 세밀하게 분석할 수 있어 본 문제에 적합합니다.

모델 구조는 다음과 같습니다.

입력 텍스트는 토크나이징 후 패딩

최종 레이어의 [CLS] 토큰을 이진 분류기에 전달

Sigmoid를 통해 AI 생성 여부를 예측합니다.

🔹 학습 방법 및 하이퍼파라미터 설정
파인튜닝 전략: Pretrained DeBERTa 위에 Linear + Sigmoid 분류기 헤드를 추가하여 Binary Cross Entropy Loss로 학습합니다.

하이퍼파라미터:

배치 크기: 16

학습률: 2e-5

에폭 수: 3

옵티마이저: AdamW

스케줄러: Linear Warmup (warmup ratio: 0.1)

검증 방식: 학습 데이터의 20%를 검증 세트로 분할하여 성능을 모니터링하였습니다.

4. 모델 검증 및 성능 평가
🔹 사용한 성능 지표
Accuracy: 전체 데이터 중 올바르게 분류한 비율

Precision: AI로 예측된 샘플 중 실제 AI가 쓴 비율

Recall: 실제 AI 텍스트 중 모델이 탐지한 비율

F1-Score: Precision과 Recall의 조화 평균

AUROC: 분류 임계값 변화에 따른 성능을 평가하는 곡선 면적

🔹 성능 비교 표
모델	Accuracy	Precision	Recall	F1-Score	AUROC
BoW + Logistic Regression	0.813	0.792	0.770	0.781	0.865
TF-IDF + SVM	0.827	0.814	0.793	0.803	0.873
DeBERTa v3 (Fine-tuned)	0.902	0.896	0.887	0.891	0.947

🔹 ROC Curve 시각화 (예시 코드)
python
복사
편집
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='DeBERTa ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show

🔹 성능 해석 및 분석
BoW 모델은 단어 출현 빈도만 반영하여 문장 구조나 의미를 학습하지 못하였습니다.

TF-IDF + SVM 조합은 일부 개선되었으나 문맥 분석 한계가 존재하였습니다.

DeBERTa v3는 표현의 반복성, 문장 구조 다양성 등 고차원 언어 패턴을 학습하여,
전반적인 성능이 크게 향상되었습니다.

특히 Recall 지표가 높은 것은 실제 AI 생성 텍스트를 잘 탐지하였다는 의미로,
실제 교육 환경 등 실사용 사례에서 높은 유용성을 보임을 시사합니다.

