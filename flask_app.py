from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

with open('logistic_model.pkl', 'rb') as modelFile:
    logisticModel = pickle.load(modelFile)

with open('vectorizer.pkl', 'rb') as vectorizerFile:
    tfidfVectorizer = pickle.load(vectorizerFile)

stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def cleanText(rawText):
    rawText = str(rawText).lower()
    rawText = re.sub(r'[^\w\s]', '', rawText)
    words = rawText.split()
    cleanedWords = [stemmer.stem(word) for word in words if word not in stopWords]
    return ' '.join(cleanedWords)

@app.route("/")
def home():
    return "✅ Sentiment Analysis API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    requestData = request.get_json()
    reviewText = requestData.get('review', '')

    if not reviewText:
        return jsonify({"error": "No review text provided."}), 400

    cleanedReview = cleanText(reviewText)
    vectorizedReview = tfidfVectorizer.transform([cleanedReview])
    prediction = logisticModel.predict(vectorizedReview)

    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return jsonify({
        "review": reviewText,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
