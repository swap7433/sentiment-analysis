import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis App")

# Sidebar settings
st.sidebar.header("Settings")
pos_threshold = st.sidebar.slider("Positive Threshold", 0.0, 1.0, 0.1, 0.05)
neg_threshold = st.sidebar.slider("Negative Threshold", -1.0, 0.0, -0.1, 0.05)
save_history = st.sidebar.checkbox("Save analysis history", value=True)

example_texts = [
    "I love this product! It's amazing and works great.",
    "This is terrible, I hate it so much.",
    "It's okay, not bad but not great either.",
    "I do not hate this movie, but it wasn't good."
]

if 'text' not in st.session_state:
    st.session_state.text = ""
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'history' not in st.session_state:
    st.session_state.history = []

def set_example_text(example):
    st.session_state.text = example
    st.session_state.analyze_clicked = True

def clear_text():
    st.session_state.text = ""
    st.session_state.analyze_clicked = False
    st.session_state.history = []

def analyze():
    if st.session_state.text.strip():
        st.session_state.analyze_clicked = True
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

text = st.text_area("Enter text to analyze:", value=st.session_state.text, height=150, key="text")
st.button("Analyze", on_click=analyze)
st.button("Clear", on_click=clear_text)

st.markdown("---")
st.write("Try one of these example texts:")
for i, example in enumerate(example_texts):
    btn_label = example if len(example) <= 40 else example[:40] + "..."
    st.button(btn_label, key=f"example_{i}", on_click=set_example_text, args=(example,))

if st.session_state.analyze_clicked and st.session_state.text.strip():
    text = st.session_state.text

    # Language detection
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"

    if lang != "en":
        st.warning(f"⚠️ Detected language: {lang}. Sentiment analysis is optimized for English.")

    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score >= pos_threshold:
        result = 'Positive 😀'
        color = '#b6e4b6'  # light green
    elif compound_score <= neg_threshold:
        result = 'Negative 😠'
        color = '#f4a3a3'  # light red
    else:
        result = 'Neutral 😐'
        color = '#d3d3d3'  # light gray

    st.markdown(
        f"**Result:** <span style='background-color:{color}; padding:5px; border-radius:5px'>{result}</span>",
        unsafe_allow_html=True
    )

    # Save to history
    if save_history:
        st.session_state.history.append({'text': text, 'sentiment': sentiment, 'result': result})

    # Show sentiment scores & bars
    st.subheader("Sentiment Scores")
    st.write(f"Compound Score: {compound_score}")
    st.progress(sentiment['pos'])
    st.write(f"Positive: {sentiment['pos']}")
    st.progress(sentiment['neu'])
    st.write(f"Neutral: {sentiment['neu']}")
    st.progress(sentiment['neg'])
    st.write(f"Negative: {sentiment['neg']}")

    # Pie chart for sentiment distribution
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiment['pos'], sentiment['neu'], sentiment['neg']]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#b6e4b6', '#d3d3d3', '#f4a3a3'])
    ax.axis('equal')
    st.pyplot(fig)

    # Word Cloud for Positive and Negative words
    st.subheader("Word Cloud")
    positive_words = ' '.join([word for word in text.split() if sia.polarity_scores(word)['compound'] > 0])
    negative_words = ' '.join([word for word in text.split() if sia.polarity_scores(word)['compound'] < 0])

    col1, col2 = st.columns(2)
    with col1:
        st.write("Positive Words")
        if positive_words:
            wc_pos = WordCloud(background_color="white").generate(positive_words)
            st.image(wc_pos.to_array())
        else:
            st.write("No positive words detected.")

    with col2:
        st.write("Negative Words")
        if negative_words:
            wc_neg = WordCloud(background_color="white").generate(negative_words)
            st.image(wc_neg.to_array())
        else:
            st.write("No negative words detected.")

    # Detailed JSON output
    st.subheader("Detailed Sentiment Scores (JSON)")
    st.json(sentiment)

# Show analysis history
if save_history and st.session_state.history:
    st.markdown("---")
    st.subheader("Analysis History")
    for idx, entry in enumerate(reversed(st.session_state.history[-5:])):
        st.markdown(f"**Entry {len(st.session_state.history) - idx}:**")
        st.write(f"Text: {entry['text']}")
        st.write(f"Result: {entry['result']}")
        st.json(entry['sentiment'])
        st.markdown("---")

# Export history button
if save_history and st.session_state.history:
    if st.button("Export History as JSON"):
        json_data = json.dumps(st.session_state.history, indent=4)
        st.download_button(label="Download JSON", data=json_data, file_name="sentiment_history.json", mime="application/json")

import pandas as pd

st.markdown("---")
st.header("📂 Batch File Upload & Sentiment Analysis")

uploaded_file = st.file_uploader("Upload CSV/TXT file (one text per line or CSV with 'text' column)", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            st.write(f"Loaded {len(df)} rows for analysis.")
            batch_results = []
            for idx, row in df.iterrows():
                t = str(row['text'])
                score = sia.polarity_scores(t)
                batch_results.append({"text": t, "sentiment": score})

            # Show batch results
            for i, res in enumerate(batch_results):
                comp = res["sentiment"]["compound"]
                label = "Positive 😀" if comp >= pos_threshold else "Negative 😠" if comp <= neg_threshold else "Neutral 😐"
                st.write(f"**Entry {i+1}:** {label} - {res['text'][:60]}...")

            # Optionally export batch results as CSV
            if st.button("Download Batch Analysis Results as CSV"):
                batch_df = pd.DataFrame(batch_results)
                batch_df['compound'] = batch_df['sentiment'].apply(lambda x: x['compound'])
                batch_df['pos'] = batch_df['sentiment'].apply(lambda x: x['pos'])
                batch_df['neu'] = batch_df['sentiment'].apply(lambda x: x['neu'])
                batch_df['neg'] = batch_df['sentiment'].apply(lambda x: x['neg'])
                csv = batch_df.drop(columns=['sentiment']).to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV", data=csv, file_name="batch_sentiment.csv", mime='text/csv')
    else:
        # TXT file
        content = uploaded_file.read().decode('utf-8').splitlines()
        st.write(f"Loaded {len(content)} lines for analysis.")
        batch_results = []
        for i, line in enumerate(content):
            if line.strip():
                score = sia.polarity_scores(line)
                batch_results.append({"text": line, "sentiment": score})
                label = "Positive 😀" if score['compound'] >= pos_threshold else "Negative 😠" if score['compound'] <= neg_threshold else "Neutral 😐"
                st.write(f"**Entry {i+1}:** {label} - {line[:60]}...")

if st.session_state.analyze_clicked and st.session_state.text.strip():
    st.markdown("---")
    st.write("Was this sentiment analysis accurate?")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    with feedback_col1:
        if st.button("👍 Yes", key="feedback_yes"):
            st.success("Thanks for your feedback!")
    with feedback_col2:
        if st.button("👎 No", key="feedback_no"):
            st.warning("Sorry to hear that. We will improve!")
    with feedback_col3:
        if st.button("💡 Suggest Improvement", key="feedback_suggest"):
            suggestion = st.text_input("Your suggestion:")
            if suggestion:
                st.write("Thanks for your suggestion!")
st.sidebar.markdown("---")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

import random

emoji_map = {
    "Positive": ["😄", "🎉", "👍", "😊"],
    "Negative": ["😢", "😠", "👎", "😭"],
    "Neutral": ["😐", "🤔", "😶"]
}


if st.session_state.analyze_clicked and st.session_state.text.strip():
    sentiment_label = None
    if compound_score >= pos_threshold:
        sentiment_label = "Positive"
    elif compound_score <= neg_threshold:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    st.markdown("### Suggested Emoji")
    st.markdown(random.choice(emoji_map[sentiment_label]), unsafe_allow_html=True)

if st.session_state.analyze_clicked and st.session_state.text.strip():
    words = st.session_state.text.split()
    word_scores = [(w, sia.polarity_scores(w)['compound']) for w in words]

    top_positive = sorted([w for w in word_scores if w[1] > 0], key=lambda x: x[1], reverse=True)[:5]
    top_negative = sorted([w for w in word_scores if w[1] < 0], key=lambda x: x[1])[:5]

    st.subheader("Explainability: Top Contributing Words")
    pos_words = ", ".join([f"{w} ({score:.2f})" for w, score in top_positive]) or "None"
    neg_words = ", ".join([f"{w} ({score:.2f})" for w, score in top_negative]) or "None"

    st.markdown(f"**Top Positive Words:** {pos_words}")
    st.markdown(f"**Top Negative Words:** {neg_words}")
