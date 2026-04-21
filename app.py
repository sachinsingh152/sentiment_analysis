from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from inference_utils import load_inference_model, clean_text, text_to_seq, load_bot_model

app = Flask(__name__)

# Config
MODEL_PATH = '/home/sachinsingh/Desktop/main_complete/Desktop/bdalab/project_local/Sentiment/bilstm_final.pth'
BOT_MODEL_DIR = '/home/sachinsingh/Desktop/main_complete/Desktop/bdalab/project_local/my_distilbert_model'
IMAGES_DIR = '/home/sachinsingh/Desktop/main_complete/Desktop/bdalab/project_local/images'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Sentiment Model
print("Loading sentiment model and vocabulary...")
model, vocab, max_len = load_inference_model(MODEL_PATH, device=DEVICE)

# Load Bot Model
print("Loading bot detection model...")
bot_model, bot_tokenizer = load_bot_model(BOT_MODEL_DIR, device=DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    raw_text = data['text']
    cleaned = clean_text(raw_text)
    
    if not cleaned:
        return jsonify({
            'sentiment': 'Neutral', 'probability': 0.5, 'label': 0,
            'is_bot': False, 'bot_probability': 0.0
        })

    # --- Sentiment Prediction (BiLSTM) ---
    seq = text_to_seq(cleaned, vocab, max_len=max_len)
    input_tensor = torch.tensor([seq]).to(DEVICE)
    
    with torch.no_grad():
        output, weights = model(input_tensor)
        probs = F.softmax(output, dim=1)
        sentiment_prob = probs[0, 1].item()
    
    sentiment = 'Positive' if sentiment_prob > 0.5 else 'Negative'

    # --- Bot Prediction (DistilBERT) ---
    inputs = bot_tokenizer(
        raw_text, 
        max_length=64, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        bot_output = bot_model(inputs['input_ids'], inputs['attention_mask'])
        bot_probs = F.softmax(bot_output, dim=1)
        bot_prob = bot_probs[0, 1].item() # index 1 is Bot, index 0 is Human

    is_bot = bot_prob > 0.5
    bot_label = "Bot" if is_bot else "Human"

    return jsonify({
        'sentiment': sentiment,
        'probability': sentiment_prob,
        'sentiment_label': 1 if sentiment_prob > 0.5 else 0,
        'is_bot': is_bot,
        'bot_label': bot_label,
        'bot_probability': bot_prob,
        'cleaned_text': cleaned
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
