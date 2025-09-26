from transformers import pipeline
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_transformer():
    model_name = config["transformer_model"]["model_name"]
    print(f"Loading transformer model: {model_name}")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, framework="pt")
    print("Model loaded.")
    return sentiment_pipeline

if __name__ == "__main__":
    nlp = load_transformer()
    print("Running inference...")
    result = nlp("This movie was absolutely fantastic!")
    print(f"Result: {result}")
