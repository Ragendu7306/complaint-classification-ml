import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. CHECK PATH
# -----------------------------
print("Current Working Directory:", os.getcwd())

csv_path = r"C:\Users\HP\Documents\enterprise genai platform\data\complaints.csv"
print("CSV File Exists:", os.path.exists(csv_path))

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
df = pd.read_csv(csv_path, low_memory=False)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# -----------------------------
# 3. TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# -----------------------------
# 4. USE CORRECT COLUMNS
# -----------------------------
# TEXT COLUMN → narrative
df["cleaned_text"] = df["narrative"].fillna("").apply(clean_text)

# LABEL COLUMN → product
X = df["cleaned_text"]
y = df["product"]

# -----------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 7. MODEL TRAINING
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# 8. MODEL EVALUATION
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 9. SAVE MODEL & VECTORIZER
# -----------------------------
joblib.dump(model, "ml_models/complaint_model.pkl")
joblib.dump(vectorizer, "ml_models/tfidf_vectorizer.pkl")

print("\n✅ Model and Vectorizer Saved Successfully")

# -----------------------------
# 10. REAL-TIME PREDICTION
# -----------------------------
while True:
    user_text = input("\nEnter complaint text (type 'exit' to stop): ")

    if user_text.lower() == "exit":
        print("👋 Exiting")
        break

    cleaned = clean_text(user_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    print("📌 Predicted Product:", prediction[0])


