import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import re
    # Loading config
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Loading config
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_data():
    raw_path = config['data']['raw_path']
    df = pd.read_csv(raw_path)
    
    # Cleaning reviews
    df['review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive' : 1, 'negative': 0})
    
    # Splitting data
    train_df, test_df = train_test_split(df, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=df['sentiment'])
    train_df, val_df = train_test_split(
        train_df, test_size=config["data"]["val_size"], random_state=config["data"]["random_state"], stratify=train_df["sentiment"]
    )
    # Saving processed data
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    train_df.to_csv(os.path.join(config['data']['processed_dir'], config['data']['train_file']), index=False)
    val_df.to_csv(os.path.join(config['data']['processed_dir'], config['data']['val_file']), index=False)
    test_df.to_csv(os.path.join(config['data']['processed_dir'], config['data']['test_file']), index=False)

    print("Data preparation completed.")
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_data()
