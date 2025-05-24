import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis App")

# Text input
text = st.text_area("Enter text to analyze:")

# Analyze on button click
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
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

        st.markdown(f"**Result:** <span style='background-color:{color};padding:5px;border-radius:5px'>{result}</span>", unsafe_allow_html=True)
        st.json(sentiment)
