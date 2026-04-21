import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import json
import os

# Download stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Loading dataset from archive...")
DATA_PATH = 'archive/training.1600000.processed.noemoticon.csv'
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', names=columns)

# We only need a sample to get the top words, similar to the notebook
SAMPLE_SIZE = 1000000
df_sample = df.sample(SAMPLE_SIZE, random_state=42)

print("Cleaning text...")
df_sample['cleaned_text'] = df_sample['text'].apply(clean_text)

print("Building vocabulary...")
all_words = ' '.join(df_sample['cleaned_text']).split()
vocab = {word: i+2 for i, (word, count) in enumerate(Counter(all_words).most_common(5000))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

print(f"✅ Vocabulary of {len(vocab)} words saved to vocab.json")
