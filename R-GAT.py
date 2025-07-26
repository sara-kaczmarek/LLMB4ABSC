import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import spacy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GATConv
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader

class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=0.2)

    def forward(self, x, edge_index, edge_type):
        return self.gat(x, edge_index)

class RGATModel(nn.Module):
    def __init__(self, hidden_size=768, rel_vocab_size=50, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.rgat = RGATLayer(hidden_size, hidden_size, rel_vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        input_ids = data.input_ids.unsqueeze(0)
        attention_mask = data.attention_mask.unsqueeze(0)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.squeeze(0)
        graph_out = self.rgat(bert_out, data.edge_index, data.edge_type)
        cls_index = (data.input_ids == 101).nonzero(as_tuple=True)[0].item()  # Find CLS token
        cls_token = graph_out[cls_index]
        return self.classifier(self.dropout(cls_token))

nlp = spacy.load("en_core_web_sm")

def build_rel_vocab(df):
    rel_set = set()
    for sent in df['sentence']:
        doc = nlp(sent)
        rel_set.update([token.dep_ for token in doc])
    rel_set.add('self')
    return {rel: i for i, rel in enumerate(sorted(rel_set))}

def convert_to_graphs(df, rel_vocab):

    nlp = spacy.load("en_core_web_sm")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    data_list = []
    for _, row in df.iterrows():
        sent = row['sentence']
        aspect = row['aspect']
        polarity = row['polarity']

        encoded = tokenizer(sent, aspect, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        doc = nlp(sent)

        edge_index = []
        edge_type = []

        for token in doc:
            src = token.i
            tgt = token.head.i
            if src < 128 and tgt < 128:
                dep_idx = rel_vocab.get(token.dep_, 1)
                edge_index.append((src, tgt))  
                edge_index.append((tgt, src)) 
                edge_type.append(dep_idx)
                edge_type.append(dep_idx)

        for i in range(len(doc)):
            if i < 128:
                edge_index.append((i, i))
                edge_type.append(rel_vocab.get('self', 0))

        graph = GeoData(
            input_ids=encoded['input_ids'].squeeze(0),
            attention_mask=encoded['attention_mask'].squeeze(0),
            edge_index=torch.tensor(edge_index, dtype=torch.long).T,  
            edge_type=torch.tensor(edge_type, dtype=torch.long),
            y=torch.tensor(label_map[polarity]),
            num_nodes=encoded['input_ids'].size(1) 
        )
        data_list.append(graph)
    return data_list


def train(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out.unsqueeze(0), batch.y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.argmax().item())
            labels.append(batch.y.item())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

def run_rgat(df_train, df_test, epochs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rel_vocab = build_rel_vocab(pd.concat([df_train, df_test]))
    df_train_split, df_val = train_test_split(df_train, test_size=0.1, stratify=df_train['polarity'], random_state=42)

    def convert(df): return convert_to_graphs(df, rel_vocab)
    train_data = convert(df_train_split)
    val_data = convert(df_val)
    test_data = convert(df_test)

    train_loader = GeoLoader(train_data, batch_size=1, shuffle=True)
    val_loader = GeoLoader(val_data, batch_size=1)
    test_loader = GeoLoader(test_data, batch_size=1)

    model = RGATModel(hidden_size=768, rel_vocab_size=len(rel_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=df_train_split['polarity'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    best_val_acc = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = F.cross_entropy(output.unsqueeze(0), batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    print(f"\nBest Val Acc = {best_val_acc:.4f}")
    model.load_state_dict(best_model)

    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Test Acc = {test_acc:.4f} | Test Macro-F1 = {test_f1:.4f}")
    return model
