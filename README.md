# AI 생성 콘텐츠 진위 판별 시스템 개발
---
전남대학교 소중단 디지털 경진대회 참가 프로젝트입니다.  
이 프로젝트는 Hugging Face의 transformer 모델(team-lucid/deberta-v3-base-korean)을 파인튜닝하여 AI와 사람이 쓴 텍스트를 분류하는 이진 분류 문제를 해결하는 것을 목표로 합니다.

## 1. Introduction
---
본 프로젝트는 전남대학교 소중단 디지털 경진대회 참가를 위해 진행되었습니다. Python, transformers, datasets, scikit-learn 등의 라이브러리를 활용하여 한국어 텍스트가 AI에 의해 작성되었는지 인간이 작성했는지를 분류하는 모델을 개발합니다. Hugging Face의 team-lucid/deberta-v3-base-korean 사전학습 모델을 기반으로 파인튜닝을 진행하며, 데이터 전처리, 모델 학습, 평가, 그리고 실사용을 위한 배포까지의 전 과정을 다룹니다.
