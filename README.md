# 💬 Sentiment Analysis Web App (Streamlit + NLTK)

This is a simple web-based Sentiment Analysis application built with **Streamlit** and **NLTK's VADER** (Valence Aware Dictionary and sEntiment Reasoner).

Users can input any English text and receive an instant sentiment classification as **Positive**, **Negative**, or **Neutral**, along with sentiment scores.

---

## 🚀 Features

- Real-time sentiment analysis using NLTK's `SentimentIntensityAnalyzer`
- Clean and interactive UI with Streamlit
- Text input validation (warns if empty)
- Color-coded sentiment results
- JSON view of the sentiment score breakdown

---

## 🛠️ Installation & Setup

### 1. Clone the repository:

```bash
git clone https://github.com/swap7433/sentiment-analysis.git
cd sentiment-analysis


# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


**## 🛠️ RUN code locally**
pip install -r requirements.txt
pip install streamlit nltk
streamlit run app.py


sentiment-analysis/
├── app.py                # Main Streamlit app
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation




