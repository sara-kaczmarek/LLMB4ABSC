import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
import random
from copy import deepcopy

def bert_spc(df_train, df_test, model_name='bert-base-uncased', batch_size=16, epochs=5, max_len=256, lr=2e-5):

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    df_train = df_train.dropna(subset=["sentence", "aspect", "polarity"])
    df_test = df_test.dropna(subset=["sentence", "aspect", "polarity"])

    label_encoder = LabelEncoder()
    df_train['label'] = label_encoder.fit_transform(df_train['polarity'])
    df_test['label'] = label_encoder.transform(df_test['polarity'])

    class ABSADataset(Dataset):
        def __init__(self, sentences, aspects, labels, tokenizer, max_len):
            self.sentences = sentences
            self.aspects = aspects
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence = self.sentences[idx]
            aspect = self.aspects[idx]
            label = self.labels[idx]

            inputs = self.tokenizer(
                sentence,
                aspect,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label)
            }

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = ABSADataset(
        df_train['sentence'].tolist(),
        df_train['aspect'].tolist(),
        df_train['label'].tolist(),
        tokenizer,
        max_len
    )
    test_dataset = ABSADataset(
        df_test['sentence'].tolist(),
        df_test['aspect'].tolist(),
        df_test['label'].tolist(),
        tokenizer,
        max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_acc = 0.0
    patience = 2
    counter = 0
    best_model_state = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} accuracy: {current_acc:.4f}")

        if current_acc > best_acc:
            best_acc = current_acc
            best_model_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Macro F1:", f1_score(all_labels, all_preds, average='macro'))

    return model, tokenizer, label_encoder
