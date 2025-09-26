# IMDb Sentiment Analysis

A full-stack machine learning project for sentiment analysis on IMDb movie reviews. Includes data processing, baseline and transformer models, evaluation, and a FastAPI deployment API.

## Features
- **Data Pipeline:** Cleans and splits raw IMDb data into train/val/test sets.
- **Baseline Model:** TF-IDF + Logistic Regression for fast, reliable sentiment predictions.
- **Transformer Model:** HuggingFace DistilBERT pipeline for deep learning-based sentiment analysis.
- **Evaluation:** Compare model performance with classification metrics.
- **API:** FastAPI endpoints for serving predictions from both models.
- **Utilities:** Logging and database support for predictions.

## Project Structure
```
data/
  raw/                # Original IMDB Dataset
  processed/          # Cleaned train/val/test splits
notebooks/
  01_eda.ipynb        # Exploratory Data Analysis
src/
  data_prep.py        # Data cleaning & splitting
  baseline_model.py   # TF-IDF + Logistic Regression
  transformer_model.py# HuggingFace pipeline
  evaluate.py         # Model evaluation
  utils.py            # Helper functions
api/
  main.py             # FastAPI app
  requirements.txt    # API dependencies
configs/
  config.yaml         # Project configs
README.md
requirements.txt      # Global dependencies
```

## Quickstart
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   ```
   python src/data_prep.py
   ```
3. **Train baseline model:**
   ```
   python src/baseline_model.py
   ```
4. **Test transformer model:**
   ```
   python src/transformer_model.py
   ```
5. **Evaluate models:**
   ```
   python src/evaluate.py
   ```
6. **Run API:**
   ```
   uvicorn api.main:app --reload
   ```

## API Endpoints
- `POST /predict_baseline/` — Predict sentiment using baseline model
- `POST /predict_transformer/` — Predict sentiment using transformer model

## Weak Points & Recommendations
- **Large Dataset:** `IMDB Dataset.csv` is >50MB. Use [Git LFS](https://git-lfs.github.com) for better versioning.
- **Transformer Model:** DistilBERT is untrained for this task; fine-tune for better accuracy.
- **Testing:** Add unit tests in `tests/` for robust validation.
- **Security:** Add input validation and error handling in API.
- **Database Logging:** `utils.py` supports SQLite logging, but integration with API is recommended.
- **Requirements:** Ensure `api/requirements.txt` is populated for deployment.

## Authors
- Ghulam Haider

## License
MIT
