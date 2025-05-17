import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 샘플 데이터셋 (AI vs Human)
data = {
    'text': [
        '이 글은 인공지능이 작성했습니다.',
        '오늘 날씨가 참 좋네요.',
        'AI가 생성한 문장입니다.',
        '저는 사람이 쓴 글입니다.',
        '이 문장은 기계가 썼어요.',
        '점심 뭐 먹지 고민돼요.',
        '이것은 AI가 만든 텍스트입니다.',
        '주말에 영화 볼래요?',
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: AI, 0: Human
}
dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained('team-lucid/deberta-v3-base-korean')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained('team-lucid/deberta-v3-base-korean', num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./results/logs',
    fp16=True,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./results/final_model')
print('학습 및 저장 완료!')
