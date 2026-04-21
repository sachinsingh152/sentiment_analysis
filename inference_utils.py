import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import re
import nltk
from nltk.corpus import stopwords
import json
import os

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
KEEP_WORDS = {'no', 'not', 'nor', 'never', 'neither', "n't", 'nothing',
               'nobody', 'nowhere', 'none', 'cannot'}
STOP_WORDS = STOP_WORDS - KEEP_WORDS

# Emoticon mapping
EMOTICONS = {
    ':)': ' happy ', ':-)': ' happy ', '=)': ' happy ', ':D': ' happy ',
    ':(': ' sad ', ':-(': ' sad ', ":'(": ' sad ',
    ';)': ' wink ', '<3': ' love ', '</3': ' heartbroken ',
    ':P': ' playful ', ':p': ' playful ', ':O': ' surprised ',
    ':@': ' angry ', '>:(': ' angry '
}

# Common Twitter contractions
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "it's": "it is",
    "i'm": "i am", "i've": "i have", "i'll": "i will",
    "i'd": "i would", "you're": "you are", "we're": "we are",
    "they're": "they are", "he's": "he is", "she's": "he is"
}

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_proj = nn.Linear(emb_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x, lengths=None):
        emb = self.dropout(self.emb_proj(self.embedding(x)))
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(emb)
        out = self.layer_norm(out)
        out = self.dropout(out)
        attn_w = torch.softmax(self.attn(out), dim=1)
        attn_ctx = (attn_w * out).sum(dim=1)
        max_pool = out.max(dim=1).values
        avg_pool = out.mean(dim=1)
        combined = torch.cat([attn_ctx, max_pool, avg_pool], dim=-1)
        logits = self.classifier(combined)
        return logits, attn_w

def clean_text(text):
    text = str(text).lower()
    for emot, repl in EMOTICONS.items():
        text = text.replace(emot, repl)
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)

def text_to_seq(text, vocab, max_len=50):
    tokens = text.split()[:max_len]
    seq = [vocab.get(t, 1) for t in tokens]
    return seq

def load_inference_model(model_path, vocab_path=None, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        vocab = checkpoint.get('vocab')
        max_len = checkpoint.get('max_len', 50)
        hidden_dim = checkpoint.get('hidden_dim', 256)
        emb_dim = checkpoint.get('emb_dim', 100)
        num_layers = checkpoint.get('num_layers', 3)
    else:
        # Fallback for old/other format
        state_dict = checkpoint
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        max_len = 50
        hidden_dim = 256
        emb_dim = 100
        num_layers = 3

    vocab_size = len(vocab)
    model = BiLSTMSentiment(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.4
    )
    
    # Adapt state dict if necessary (sometimes Kaggle saves with prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, vocab, max_len

# --- Bot Detection Logic ---
from transformers import DistilBertTokenizer, DistilBertModel

class BotModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_dir)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0]   # CLS token
        return self.fc(x)

def load_bot_model(model_dir, device='cpu'):
    # Load tokenizer from the directory
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    # Initialize model with base BERT from same directory
    model = BotModel(model_dir)
    
    # Load the trained linear head weights
    state_dict_path = os.path.join(model_dir, "full_model_state.pth")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model, tokenizer
