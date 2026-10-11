
import json
from pathlib import Path

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


# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


# 1. Load IMDb dataset
print("Downloading/loading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

# Shuffle before selecting so we get both classes.
train_data = dataset["train"].shuffle(seed=42).select(
    range(10000)
)

test_data = dataset["test"].shuffle(seed=42).select(
    range(2500)
)

X_train = list(train_data["text"])
y_train = list(train_data["label"])

X_test = list(test_data["text"])
y_test = list(test_data["label"])

print("Training labels:", set(y_train))
print("Label counts:", {label: y_train.count(label) for label in set(y_train)})

print(f"Training reviews: {len(X_train)}")
print(f"Testing reviews: {len(X_test)}")


# 2. Build the ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])


# 3. Train
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)


# 4. Evaluate on unseen test reviews
print("Evaluating model...")

predictions = model.predict(X_test)

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "dataset": "IMDb",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "accuracy": round(
        float(accuracy_score(y_test, predictions)), 4
    ),
    "precision": round(
        float(precision_score(y_test, predictions)), 4
    ),
    "recall": round(
        float(recall_score(y_test, predictions)), 4
    ),
    "f1_score": round(
        float(f1_score(y_test, predictions)), 4
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}


# 5. Save model and metrics
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# 6. Test a sample review
sample = (
    "The movie was wonderful, entertaining, "
    "and beautifully written."
)

prediction = int(model.predict([sample])[0])
probabilities = model.predict_proba([sample])[0]

print("\nSample prediction:")
print("Sentiment:", "Positive" if prediction == 1 else "Negative")
print(f"Confidence: {max(probabilities) * 100:.2f}%")

print("\nEvaluation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")