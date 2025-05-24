import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis App")

# Text input
text = st.text_area("Enter text to analyze:")

# Analyze button
if st.button("Analyze"):
    if not text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        sentiment = sia.polarity_scores(text)
        compound_score = sentiment['compound']

        if compound_score > 0.1 and not ('not hate' in text.lower()):
            result = 'Positive 😀'
            color = 'lightgreen'
        elif compound_score < -0.1:
            result = 'Negative 😠'
            color = 'salmon'
        else:
            result = 'Neutral 😐'
            color = 'lightgray'

        st.markdown(
            f"**Result:** <span style='background-color:{color}; padding:5px; border-radius:5px'>{result}</span>",
            unsafe_allow_html=True
        )
        st.json(sentiment)
