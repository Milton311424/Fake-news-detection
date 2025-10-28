from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ✅ Use absolute paths for Azure
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../model/fake_news_model.pkl")
vectorizer_path = os.path.join(base_dir, "../model/vectorizer.pkl")

# ✅ Load model & vectorizer safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    transformed_input = vectorizer.transform([text])
    prediction = model.predict(transformed_input)[0]
    return render_template('index.html', prediction_text=f"The news is predicted to be: {prediction}")

if __name__ == "__main__":
    # ✅ Important for Azure compatibility
    app.run(host='0.0.0.0', port=8000, debug=True)
