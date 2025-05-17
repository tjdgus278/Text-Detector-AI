import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class CRAFT:
    SEED = 1
    EPOCH = 3
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 16
    SAVE_STRATEGY = 'epoch'
    EVALUATION_STRATEGY = 'epoch'
    METRIC = 'f1'
    EXP_NUM = 1
    model = 'team-lucid/deberta-v3-base-korean'
    train_df_path = './df_final_train_v1.csv'
    valid_df_path = './df_final_valid_v1.csv'
    test_df_v1_path = './df_final_test_v1.csv'
    test_df_v2_path = './df_final_test_v2.csv'
    model_output_path = f'./results_{EXP_NUM}'
    test_v1_inference_path = f'test_v1_predictions_{EXP_NUM}.csv'
    test_v2_inference_path = f'test_v2_predictions_{EXP_NUM}.csv'

# 데이터 로드 및 셔플
train_df = pd.read_csv(CRAFT.train_df_path)
valid_df = pd.read_csv(CRAFT.valid_df_path)
test_df_v1 = pd.read_csv(CRAFT.test_df_v1_path)
test_df_v2 = pd.read_csv(CRAFT.test_df_v2_path)

train_df = train_df.sample(frac=1, random_state=CRAFT.SEED).reset_index(drop=True)
valid_df = valid_df.sample(frac=1, random_state=CRAFT.SEED).reset_index(drop=True)
test_df_v1 = test_df_v1.sample(frac=1, random_state=CRAFT.SEED).reset_index(drop=True)
test_df_v2 = test_df_v2.sample(frac=1, random_state=CRAFT.SEED).reset_index(drop=True)

def preprocess_data(df):
    return Dataset.from_pandas(df[['text', 'label']])

train_dataset = preprocess_data(train_df)
valid_dataset = preprocess_data(valid_df)
test_v1_dataset = preprocess_data(test_df_v1)
test_v2_dataset = preprocess_data(test_df_v2)

model_name = CRAFT.model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_v1_dataset = test_v1_dataset.map(tokenize_function, batched=True)
test_v2_dataset = test_v2_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
valid_dataset = valid_dataset.rename_column("label", "labels")
test_v1_dataset = test_v1_dataset.rename_column("label", "labels")
test_v2_dataset = test_v2_dataset.rename_column("label", "labels")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_v1_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_v2_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 데이터 컬럼 체크 및 예외 처리
REQUIRED_COLUMNS = {'text', 'label'}
def check_columns(df, name):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} 데이터프레임에 다음 컬럼이 없습니다: {missing}")

check_columns(train_df, 'train_df')
check_columns(valid_df, 'valid_df')
check_columns(test_df_v1, 'test_df_v1')
check_columns(test_df_v2, 'test_df_v2')

# seed 고정 함수 추가
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(CRAFT.SEED)

# 결과 폴더 생성
os.makedirs(CRAFT.model_output_path, exist_ok=True)

# fp16 옵션 자동 조정 (CUDA가 없으면 False)
import transformers
if not torch.cuda.is_available():
    fp16_flag = False
else:
    fp16_flag = True

# compute_metrics_for_test와 compute_metrics 통합

def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer용 래퍼

def compute_metrics_trainer(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return compute_metrics(labels, preds)

training_args = TrainingArguments(
    output_dir=CRAFT.model_output_path,
    evaluation_strategy=CRAFT.EVALUATION_STRATEGY,
    save_strategy=CRAFT.SAVE_STRATEGY,
    per_device_train_batch_size=CRAFT.BATCH_SIZE,
    per_device_eval_batch_size=CRAFT.BATCH_SIZE,
    num_train_epochs=CRAFT.EPOCH,
    weight_decay=CRAFT.WEIGHT_DECAY,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model=CRAFT.METRIC,
    learning_rate=CRAFT.LR,
    greater_is_better=True,
    seed=CRAFT.SEED,
    fp16=fp16_flag
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_trainer
)

trainer.train()

eval_results = trainer.evaluate()
print("Validation Results:\n", eval_results)

# test v1

test_v1_predictions = trainer.predict(test_v1_dataset)
test_v1_preds = torch.argmax(torch.tensor(test_v1_predictions.predictions), dim=-1).numpy()
test_v1_labels = test_v1_predictions.label_ids
test_metrics = compute_metrics(test_v1_labels, test_v1_preds)
print("Test v1 Inference Results:", test_metrics)
test_df_v1['predicted_label'] = test_v1_preds
test_df_v1.to_csv(CRAFT.test_v1_inference_path, index=False)

# test v2
test_v2_predictions = trainer.predict(test_v2_dataset)
test_v2_preds = torch.argmax(torch.tensor(test_v2_predictions.predictions), dim=-1).numpy()
test_v2_labels = test_v2_predictions.label_ids
test_metrics = compute_metrics(test_v2_labels, test_v2_preds)
print("Test v2 Inference Results:", test_metrics)
test_df_v2['predicted_label'] = test_v2_preds
test_df_v2.to_csv(CRAFT.test_v2_inference_path, index=False)
