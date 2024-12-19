import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json

def normalize(x):
    return x.replace(',', '').replace('.', '').replace('"', '').replace('!', '').replace('\n', '').lower()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)

    # Метрики
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    # AUC ROC
    try:
        auc = roc_auc_score(labels, logits[:, 1])  # Используется вероятность для класса 1
    except ValueError:
        auc = None  # AUC не вычисляется, если только один класс

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc
    }

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



df = pd.read_csv('updated_data.csv', encoding='utf-8', delimiter=';')
dv = pd.read_csv('updated_data_valid.csv', encoding='utf-8', delimiter=';')

texts = df['text'].apply(lambda x: normalize(x)).tolist()
labels = df['label'].apply(lambda x: 1 if x == 'commercial' else 0).tolist()

textsV = dv['text'].apply(lambda x: normalize(x)).tolist()
labelsV = dv['label'].apply(lambda x: 1 if x == 'commercial' else 0).tolist()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.config.pad_token_id = model.config.eos_token_id


dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=128)
dataset_eval = TextClassificationDataset(textsV, labelsV, tokenizer, max_length=128)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

training_args = TrainingArguments(
    output_dir='./results5',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=10,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()

metrics = trainer.state.log_history

# Сохранение метрик в JSON-файл
with open('metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

model.save_pretrained('./commerce_classifier')
tokenizer.save_pretrained('./commerce_classifier')
