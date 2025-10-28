from flask import Flask, render_template, request
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# ✅ Define base directory (important for Azure deployment)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Load model and vectorizer using absolute paths
model_path = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# Load trained model and TF-IDF vectorizer
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    transformed_input = vectorizer.transform([text])
    prediction = model.predict(transformed_input)[0]

    return render_template('index.html', prediction_text=f"The news is predicted to be: {prediction}")

# ✅ Azure entry point
if __name__ == "__main__":
    # Use host='0.0.0.0' for Azure App Service compatibility
    app.run(host='0.0.0.0', port=5000, debug=True)
