import torch
import joblib
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nltk.tokenize import word_tokenize
import re
import torch.nn as nn

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

# --- Initialize FastAPI app ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Load the model and vocabulary ---
try:
    vocab = torch.load('vocab.pth')  # Load vocabulary
    model = SentimentClassifier(len(vocab), 100, 256, 2)  # Define model architecture
    model.load_state_dict(torch.load('sentiment_model_with_negations.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    print("Model and Vocabulary loaded successfully!")
except Exception as e:
    print(f"Error loading model or vocab: {e}")
    raise RuntimeError("Model or vocabulary not found.")

# --- Preprocessing function ---
def preprocess_text_with_vocab(text, vocab, max_len=100):
    """
    Preprocess text using the vocabulary. Tokenize, clean, and pad/clip the text.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())  # Clean text
    tokens = word_tokenize(text)  # Tokenize
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]  # Convert tokens to indices
    return sequence[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(sequence))  # Pad or truncate

# --- Route: Display input form ---
@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    """
    Render the input form (form.html) for sentiment analysis.
    """
    return templates.TemplateResponse("form.html", {"request": request})

# --- Route: Process the input and predict sentiment ---
@app.post("/submit", response_class=HTMLResponse)
def submit(request: Request, text: str = Form(...)):
    """
    Handle form submission, preprocess input text, and return sentiment prediction.
    """
    try:
        # Preprocess the input text
        sequence = preprocess_text_with_vocab(text, vocab)
        vectorized_review = torch.tensor([sequence], dtype=torch.long)

        # Predict sentiment
        with torch.no_grad():
            output = model(vectorized_review)
            _, predicted_class = torch.max(output, 1)  # Get the class with the highest score

        # Map predicted class to sentiment label
        sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

        # Render result.html with the prediction
        return templates.TemplateResponse("result.html", {
            "request": request,
            "text": text,
            "sentiment": sentiment
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
