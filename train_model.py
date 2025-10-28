import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# ✅ File paths
fake_path = "dataset/Fake.csv"
true_path = "dataset/True.csv"

# ✅ Check files exist
if not os.path.exists(fake_path) or not os.path.exists(true_path):
    raise FileNotFoundError("Please place 'Fake.csv' and 'True.csv' inside the dataset/ folder!")

print("✅ Loading datasets...")
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)
print(f"Fake news rows: {len(fake_df)}, True news rows: {len(true_df)}")

# ✅ Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# ✅ Combine both datasets
data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

# ✅ Clean text
data["text"] = data["title"].fillna('') + " " + data["text"].fillna('')
data = data.dropna(subset=["text", "label"])
data = data[data["text"].str.strip() != ""]

# ✅ Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ✅ TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# ✅ Model Training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# ✅ Evaluate Accuracy
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {round(accuracy * 100, 2)}%")

# ✅ Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("💾 Model and vectorizer saved successfully in 'model/' folder!")
