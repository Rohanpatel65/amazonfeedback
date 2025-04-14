from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return "API is up! Use the `/predict` endpoint with POST."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
