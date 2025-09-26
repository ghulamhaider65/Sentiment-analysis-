import pandas as pd
import joblib, yaml, os
from sklearn.metrics import classification_report
from transformers import pipeline

# Loading config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def evaluate_models(limit_samples=500):
    processed_dir = config['data']['processed_dir']
    val_df = pd.read_csv(os.path.join(processed_dir, config['data']['val_file']))
    print(f"Validation shape: {val_df.shape}")

    # Baseline model
    vectorizer, clf = joblib.load("baseline_model.pkl")
    X_val = vectorizer.transform(val_df["review"][:limit_samples])
    baseline_preds = clf.predict(X_val)
    print("Baseline Model Evaluation:")
    print(classification_report(val_df["sentiment"][:limit_samples], baseline_preds))

    # Transformer model
    nlp = pipeline("sentiment-analysis", model=config["transformer_model"]["model_name"], framework="pt", truncation=True, max_length=512)
    sample_reviews = val_df["review"].tolist()[:limit_samples]
    transformer_results = nlp(sample_reviews)
    # Convert labels (LABEL_1 is positive, LABEL_0 is negative for this model)
    transformer_preds = [1 if r['label'] == 'LABEL_1' else 0 for r in transformer_results]
    print("Transformer Model Evaluation:")
    print(classification_report(val_df["sentiment"][:limit_samples], transformer_preds))

if __name__ == "__main__":
    evaluate_models()