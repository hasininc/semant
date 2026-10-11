
from pathlib import Path
import json
import joblib

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# File paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sentiment_char_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "char_metrics.json"

# 1. Load the same IMDb dataset and same split as the baseline
print("Loading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"].shuffle(seed=42).select(range(10000))
test_data = dataset["test"].shuffle(seed=42).select(range(2500))

X_train = list(train_data["text"])
y_train = list(train_data["label"])

X_test = list(test_data["text"])
y_test = list(test_data["label"])

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 2. Build the character-level model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

# 3. Train
print("\nTraining character-level Logistic Regression...")
model.fit(X_train, y_train)

# 4. Evaluate
print("Evaluating model...")

predictions = model.predict(X_test)

metrics = {
    "model": "Character TF-IDF + Logistic Regression",
    "dataset": "IMDb",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
}

# 5. Save model and metrics
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

# 6. Display results
print("\n===== Character Model Results =====")
for key, value in metrics.items():
    print(f"{key}: {value}")

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")