import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")
vocab = word_vectors.key_to_index
embedding_dim = word_vectors.vector_size

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.sentences = df['sentence'].tolist()
        self.aspects = df['aspect'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence_tokens = self.tokenizer(self.sentences[idx])
        aspect_tokens = self.tokenizer(self.aspects[idx])
        sentence_ids = [vocab.get(w, 0) for w in sentence_tokens]
        aspect_ids = [vocab.get(w, 0) for w in aspect_tokens]
        label = self.labels[idx]
        return torch.tensor(sentence_ids), torch.tensor(aspect_ids), torch.tensor(label)

def collate_fn(batch):
    sentences, aspects, labels = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True)
    padded_aspects = pad_sequence(aspects, batch_first=True)
    return padded_sentences, padded_aspects, torch.tensor(labels)

class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes):
        super(ATAE_LSTM, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, aspect):
        x_embed = self.embed(x)
        aspect_embed = self.embed(aspect)
        aspect_mean = aspect_embed.mean(dim=1, keepdim=True)
        aspect_repeated = aspect_mean.repeat(1, x.size(1), 1)
        x_concat = torch.cat((x_embed, aspect_repeated), dim=2)
        h_lstm, _ = self.lstm(x_concat)
        attn_weights = torch.softmax(self.attention(h_lstm).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(h_lstm * attn_weights, dim=1)
        output = self.fc(context)
        return output

def run_atae_lstm(df_train, df_test, batch_size=32, epochs=5, max_len=128, hidden_dim=100):
    df_train = df_train.dropna(subset=['sentence', 'aspect', 'polarity'])
    df_test = df_test.dropna(subset=['sentence', 'aspect', 'polarity'])

    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['polarity'])
    df_test['label'] = le.transform(df_test['polarity'])

    tokenizer = lambda x: x.lower().split()
    train_data = ABSADataset(df_train, tokenizer, max_len)
    test_data = ABSADataset(df_test, tokenizer, max_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    embedding_matrix = np.zeros((len(vocab)+1, embedding_dim))
    for word, idx in vocab.items():
        embedding_matrix[idx] = word_vectors[word]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATAE_LSTM(embedding_matrix, hidden_dim, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for sentences, aspects, labels in loop:
            sentences, aspects, labels = sentences.to(device), aspects.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences, aspects)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for sentences, aspects, labels in test_loader:
            sentences, aspects = sentences.to(device), aspects.to(device)
            outputs = model(sentences, aspects)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(predictions)
            true.extend(labels.numpy())

    print(classification_report(true, preds, target_names=le.classes_))
    print("Accuracy:", accuracy_score(true, preds))
    print("Macro F1:", f1_score(true, preds, average='macro'))

    return model
