from fastapi import FastAPI
import joblib
from transformers import pipeline

app = FastAPI()

# Load baseline model
vectorizer, clf = joblib.load("baseline_model.pkl")

# Load transformer
transformer = pipeline("sentiment-analysis", model="distilbert-base-uncased", framework="pt", truncation=True, max_length=512)

@app.get("/")
def home():
    return {"message": "IMDb Sentiment API is running!"}

@app.post("/predict_baseline/")
def predict_baseline(review: str):
    X = vectorizer.transform([review])
    pred = clf.predict(X)[0]
    sentiment_label = "positive" if pred == 1 else "negative"
    return {"review": review, "sentiment": sentiment_label, "prediction": int(pred)}

@app.post("/predict_transformer/")
def predict_transformer(review: str):
    result = transformer(review)[0]
    # Convert LABEL_1 to positive, LABEL_0 to negative
    sentiment_label = "positive" if result["label"] == "LABEL_1" else "negative"
    return {"review": review, "sentiment": sentiment_label, "score": result["score"], "raw_label": result["label"]}
