"""Unified model inference engine for SemanticShield.

Supports Model A (TF-IDF), Model B (DistilBERT), and Model C (DistilBERT Robust).
Handles PyTorch MPS/CPU device placement, lazy loading, and consistent response formats.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Device selection
if torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
else:
    TORCH_DEVICE = torch.device("cpu")

# Model Registry
_loaded_models: Dict[str, Any] = {}
_loaded_tokenizers: Dict[str, Any] = {}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Inspects available model checkpoints on disk."""
    return {
        "tfidf": {
            "name": "TF-IDF + Logistic Regression",
            "type": "Word-level N-gram Baseline",
            "available": (MODELS_DIR / "sentiment_pipeline.joblib").exists(),
            "path": str(MODELS_DIR / "sentiment_pipeline.joblib"),
            "is_transformer": False,
        },
        "distilbert": {
            "name": "DistilBERT (Fine-tuned)",
            "type": "Transformer Classifier (Model B)",
            "available": (MODELS_DIR / "distilbert_sentiment").exists(),
            "path": str(MODELS_DIR / "distilbert_sentiment"),
            "is_transformer": True,
        },
        "distilbert_robust": {
            "name": "DistilBERT (Robustness-Enhanced)",
            "type": "Augmented Transformer (Model C)",
            "available": (MODELS_DIR / "distilbert_robust").exists(),
            "path": str(MODELS_DIR / "distilbert_robust"),
            "is_transformer": True,
        },
    }


def load_model(model_id: str):
    """Loads a model into memory cache if not already loaded."""
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    models_info = get_available_models()
    if model_id not in models_info:
        raise ValueError(f"Unknown model_id: '{model_id}'. Available: {list(models_info.keys())}")

    info = models_info[model_id]

    if not info["is_transformer"]:
        path = Path(info["path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}. Run python train.py first.")
        model = joblib.load(path)
        _loaded_models[model_id] = model
        return model

    # Transformer models
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path(info["path"])
    if not model_path.exists():
        # Fallback to base pretrained checkpoint if fine-tuned checkpoint not yet available
        checkpoint_name = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"Notice: Checkpoint {model_path} not found. Loading pretrained fallback: {checkpoint_name}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name).to(TORCH_DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(TORCH_DEVICE)

    model.eval()
    _loaded_tokenizers[model_id] = tokenizer
    _loaded_models[model_id] = model
    return model


def predict(text: str, model_id: str = "tfidf") -> Dict[str, Any]:
    """Generates sentiment prediction for input text using selected model."""
    text_clean = text.strip()
    if not text_clean:
        raise ValueError("Text cannot be empty.")

    model = load_model(model_id)
    models_info = get_available_models()
    model_name = models_info[model_id]["name"]

    if not models_info[model_id]["is_transformer"]:
        prediction = int(model.predict([text_clean])[0])
        probabilities = model.predict_proba([text_clean])[0]

        class_probabilities = {
            int(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

        sentiment = "Positive" if prediction == 1 else "Negative"
        neg_prob = round(class_probabilities.get(0, 0.0) * 100, 2)
        pos_prob = round(class_probabilities.get(1, 0.0) * 100, 2)
        conf = pos_prob if sentiment == "Positive" else neg_prob

        return {
            "text": text_clean,
            "sentiment": sentiment,
            "confidence": conf,
            "probabilities": {
                "negative": neg_prob,
                "positive": pos_prob,
            },
            "model": model_name,
            "model_id": model_id,
        }

    # Transformer inference
    tokenizer = _loaded_tokenizers[model_id]
    with torch.no_grad():
        inputs = tokenizer(
            text_clean,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(TORCH_DEVICE)

        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

    # DistilBERT binary classification: index 0 = Negative, index 1 = Positive
    neg_prob = round(float(probs[0]) * 100, 2)
    pos_prob = round(float(probs[1]) * 100, 2)
    sentiment = "Positive" if pos_prob >= neg_prob else "Negative"
    confidence = pos_prob if sentiment == "Positive" else neg_prob

    return {
        "text": text_clean,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": neg_prob,
            "positive": pos_prob,
        },
        "model": model_name,
        "model_id": model_id,
    }
