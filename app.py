from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize Flask app and SentimentIntensityAnalyzer
app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score > 0.1 and not ('not hate' in text.lower()):
        result = 'Positive'
        color_class = 'positive'
    elif compound_score < -0.1:
        result = 'Negative'
        color_class = 'negative'
    else:
        result = 'Neutral'
        color_class = 'neutral'

    return render_template('index.html', text=text, result=result, sentiment=sentiment, color_class=color_class)


if __name__ == '__main__':
    app.run(debug=True)
