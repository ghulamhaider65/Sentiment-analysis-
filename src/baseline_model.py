import pandas as pd
import yaml
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Loading config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_baseline():
    processed_dir = config["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(processed_dir, config["data"]["train_file"]))
    val_df = pd.read_csv(os.path.join(processed_dir, config["data"]["val_file"]))
    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=config["baseline_model"]["vectorizer"]["max_features"],
                                 ngram_range=tuple(config["baseline_model"]["vectorizer"]["ngram_range"]))
    X_train = vectorizer.fit_transform(train_df["review"])
    X_val = vectorizer.transform(val_df["review"])
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # Train model
    clf = LogisticRegression(max_iter=config["baseline_model"]["classifier"]["max_iter"],
                             C=config["baseline_model"]["classifier"]["C"])
    clf.fit(X_train, train_df["sentiment"])
    print("Model training completed.")

    # Evaluation
    preds = clf.predict(X_val)
    print(classification_report(val_df["sentiment"], preds))

    # Save model + vectorizer
    joblib.dump((vectorizer, clf), "baseline_model.pkl")
    print("Baseline model saved")

if __name__ == "__main__":
    train_baseline()
