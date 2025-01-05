import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re

# Download NLTK Data
nltk.download('stopwords')
nltk.download('punkt')

class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- Attention Layer ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        scores = self.attention(lstm_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weighted_sum = (lstm_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return weighted_sum

# --- LSTM-Based Model with Attention ---
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_outputs, _ = self.lstm(x)
        attention_output = self.attention(lstm_outputs)
        out = self.fc(attention_output)
        return out

# --- Tokenization and Padding ---
def tokenize_and_pad(texts, vocab=None, max_len=100):
    def preprocess(text):
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        return word_tokenize(text)

    tokenized_texts = [preprocess(text) for text in texts]

    if vocab is None:
        all_tokens = [token for text in tokenized_texts for token in text]
        token_counts = Counter(all_tokens)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(token_counts.most_common())}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1

    sequences = [
        [vocab.get(token, vocab["<UNK>"]) for token in text] for text in tokenized_texts
    ]
    padded_sequences = [
        seq[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(seq)) for seq in sequences
    ]
    return padded_sequences, vocab

# --- Negation Handling ---
def preprocess_text_with_negations(text):
    negation_words = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "doesn't"}
    tokens = word_tokenize(text.lower())
    negation_count = sum(1 for token in tokens if token in negation_words)
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens), negation_count

# --- Main Script ---
try:
    df = pd.read_csv('IMDB Dataset.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    stop_words = set(stopwords.words('english'))

    df['preprocessed_data'] = df['review'].apply(preprocess_text_with_negations)
    df['cleaned_text'] = df['preprocessed_data'].apply(lambda x: x[0])
    df['negation_count'] = df['preprocessed_data'].apply(lambda x: x[1])

    texts = df['cleaned_text'].tolist()
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
    padded_sequences, vocab = tokenize_and_pad(texts)
    X = padded_sequences
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    vocab_size = len(vocab)
    embed_dim = 100
    hidden_dim = 256
    output_dim = 2

    model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            features = batch['features']
            labels = batch['label']
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'sentiment_model_with_negations.pth')
    torch.save(vocab, 'vocab.pth')
    print("\nModel and Vocabulary Saved!")
except Exception as e:
    print("An error occurred:", e)
