"""Unified model inference engine for SemanticShield.

Supports:
- Word-level TF-IDF + Logistic Regression (Model A)
- Character-level TF-IDF + Logistic Regression (Model A2)
- DistilBERT Transformer Classifier (Model B)
- DistilBERT Robustness-Enhanced (Model C, when available)

Handles PyTorch MPS/CPU device placement, lazy loading, diagnostic logging,
and temperature scaling calibration (T=1.50 fitted on IMDb validation split).
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SemanticShieldInference")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Device selection: MPS for macOS Apple Silicon, CUDA for GPU, CPU fallback
if torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
else:
    TORCH_DEVICE = torch.device("cpu")

# Calibration temperature fitted on IMDb validation set (N=1000)
DISTILBERT_CALIBRATION_TEMPERATURE = 1.50

# Model Registry
_loaded_models: Dict[str, Any] = {}
_loaded_tokenizers: Dict[str, Any] = {}
_loaded_checkpoint_names: Dict[str, str] = {}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Inspects available model checkpoints on disk."""
    return {
        "tfidf": {
            "name": "TF-IDF + Logistic Regression (Word)",
            "type": "Word-level N-gram Baseline",
            "available": (MODELS_DIR / "sentiment_pipeline.joblib").exists(),
            "path": str(MODELS_DIR / "sentiment_pipeline.joblib"),
            "is_transformer": False,
        },
        "tfidf_char": {
            "name": "Character TF-IDF + Logistic Regression",
            "type": "Character N-gram Baseline (3-5 chars)",
            "available": (MODELS_DIR / "sentiment_char_pipeline.joblib").exists(),
            "path": str(MODELS_DIR / "sentiment_char_pipeline.joblib"),
            "is_transformer": False,
        },
        "distilbert": {
            "name": "DistilBERT (Transformer)",
            "type": "Transformer Classifier (Model B)",
            "available": True,
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
            raise FileNotFoundError(f"Model file not found at {path}.")
        model = joblib.load(path)
        _loaded_models[model_id] = model
        logger.info(f"Loaded Sklearn model '{model_id}' from {path}")
        return model

    # Transformer models
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path(info["path"])
    if not model_path.exists() or not (model_path / "model.safetensors").exists():
        checkpoint_name = "distilbert-base-uncased-finetuned-sst-2-english"
        logger.info(f"Checkpoint at {model_path} not found on disk. Loading pretrained checkpoint: '{checkpoint_name}' onto {TORCH_DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name).to(TORCH_DEVICE)
        _loaded_checkpoint_names[model_id] = checkpoint_name
    else:
        logger.info(f"Loading local fine-tuned checkpoint from {model_path} onto {TORCH_DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(TORCH_DEVICE)
        _loaded_checkpoint_names[model_id] = str(model_path)

    model.eval()
    logger.info(f"DistilBERT '{model_id}' config: id2label={model.config.id2label}, label2id={model.config.label2id}, eval_mode={not model.training}")
    _loaded_tokenizers[model_id] = tokenizer
    _loaded_models[model_id] = model
    return model


def predict(text: str, model_id: str = "tfidf") -> Dict[str, Any]:
    """Generates sentiment prediction for input text using selected model.
    
    Includes diagnostic logs and both raw and temperature-calibrated probabilities.
    """
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

        logger.info(f"[Sklearn {model_id}] text='{text_clean}' -> {sentiment} ({conf}%) [neg={neg_prob}%, pos={pos_prob}%]")

        return {
            "text": text_clean,
            "sentiment": sentiment,
            "confidence": conf,
            "raw_confidence": conf,
            "calibrated_confidence": conf,
            "probabilities": {
                "negative": neg_prob,
                "positive": pos_prob,
            },
            "raw_probabilities": {
                "negative": neg_prob,
                "positive": pos_prob,
            },
            "calibrated_probabilities": {
                "negative": neg_prob,
                "positive": pos_prob,
            },
            "raw_logits": None,
            "logit_margin": None,
            "temperature": 1.0,
            "model": model_name,
            "model_id": model_id,
        }

    # Transformer inference
    tokenizer = _loaded_tokenizers[model_id]
    checkpoint_id = _loaded_checkpoint_names.get(model_id, "distilbert-base-uncased-finetuned-sst-2-english")
    tokens = tokenizer.tokenize(text_clean)

    with torch.no_grad():
        inputs = tokenizer(
            text_clean,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(TORCH_DEVICE)

        outputs = model(**inputs)
        logits = outputs.logits[0].cpu()

        # Raw softmax probabilities (exactly once along class dimension)
        raw_probs = F.softmax(logits, dim=-1).numpy()

        # Calibrated softmax probabilities via temperature scaling (T=1.50)
        T = DISTILBERT_CALIBRATION_TEMPERATURE
        cal_probs = F.softmax(logits / T, dim=-1).numpy()

    # DistilBERT id2label mapping: 0 -> NEGATIVE, 1 -> POSITIVE
    pred_idx = int(torch.argmax(logits).item())
    sentiment = "Positive" if pred_idx == 1 else "Negative"

    raw_neg = round(float(raw_probs[0]) * 100, 2)
    raw_pos = round(float(raw_probs[1]) * 100, 2)
    raw_conf = raw_pos if sentiment == "Positive" else raw_neg

    cal_neg = round(float(cal_probs[0]) * 100, 2)
    cal_pos = round(float(cal_probs[1]) * 100, 2)
    cal_conf = cal_pos if sentiment == "Positive" else cal_neg

    logit_neg = round(float(logits[0]), 4)
    logit_pos = round(float(logits[1]), 4)
    margin = round(abs(logit_pos - logit_neg), 4)

    # Diagnostic logging (Item #2 in verification requirements)
    logger.info(
        f"[DistilBERT {checkpoint_id}] text='{text_clean}' tokens={tokens} "
        f"logits=[neg:{logit_neg}, pos:{logit_pos}] margin={margin} pred_idx={pred_idx} ({sentiment}) "
        f"raw_conf={raw_conf}% cal_conf={cal_conf}% (T={T})"
    )

    return {
        "text": text_clean,
        "sentiment": sentiment,
        "confidence": raw_conf,  # Preserves raw confidence for standard contract
        "raw_confidence": raw_conf,
        "calibrated_confidence": cal_conf,
        "probabilities": {
            "negative": raw_neg,
            "positive": raw_pos,
        },
        "raw_probabilities": {
            "negative": raw_neg,
            "positive": raw_pos,
        },
        "calibrated_probabilities": {
            "negative": cal_neg,
            "positive": cal_pos,
        },
        "raw_logits": [logit_neg, logit_pos],
        "logit_margin": margin,
        "temperature": T,
        "tokens": tokens,
        "checkpoint": checkpoint_id,
        "id2label": model.config.id2label,
        "model": model_name,
        "model_id": model_id,
    }
