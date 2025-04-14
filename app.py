from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        review = request.form.get("review")
        if review:
            cleaned = review.lower()
            vector = vectorizer.transform([cleaned])
            result = model.predict(vector)[0]
            prediction = "Positive 😊" if result == 1 else "Negative 😞"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
