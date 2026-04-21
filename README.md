# SENTIX: Sentiment & Bot Analysis Dashboard

SENTIX is a professional, production-grade web application designed to provide deep insights into Twitter content. It combines advanced Deep Learning architectures to perform simultaneous **Sentiment Analysis** and **Bot Detection**.

![Dashboard Mockup](static/images/screenshot_placeholder.png) <!-- Update this with an actual screenshot later -->

## 🚀 Features

- **Dual-Model Inference**:
  - **Sentiment Analysis**: Powered by a BiLSTM-Attention Neural Network trained on a large corpus of tweets.
  - **Bot Detection**: Utilizes a DistilBERT transformer model to classify whether a tweet originates from a Human or a Bot.
- **Modern Dashboard UI**: A premium "Glassmorphism" interface with responsive layouts and interactive components.
- **Visual Insights**: Integrated Exploratory Data Analysis (EDA) visualizations including:
  - Label Distributions
  - Top Words in Corpus
  - Bigram Analysis
- **Production Ready**: Flask-based backend with a robust preprocessing pipeline for real-time text cleaning and tokenization.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, PyTorch
- **Transformers**: Hugging Face DistilBERT
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6)
- **Data Science**: NLTK, Pandas, NumPy, Matplotlib

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sentix.git
   cd sentix
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Model Weights**:
   - Place `bilstm_final.pth` in the `Sentiment/` directory.
   - Place the DistilBERT model files in the `my_distilbert_model/` directory.

5. **Run the Application**:
   ```bash
   python app.py
   ```
   Access the dashboard at `http://127.0.0.1:5000`.

## 📂 Project Structure

- `app.py`: Main Flask application and API routes.
- `inference_utils.py`: Preprocessing logic and model class definitions.
- `templates/`: HTML structures.
- `static/`:
  - `css/`: Modern UI styling.
  - `js/`: Frontend interaction logic.
- `images/`: Dashboard visualization assets.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
