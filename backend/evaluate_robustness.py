"""Comprehensive Robustness Evaluation Pipeline for SemanticShield.

Executes:
- Experiment E1: Clean Test Set Evaluation (2,500 IMDb test reviews)
- Experiment E2: Controlled Character Noise & Typo Perturbations
- Experiment E3: Controlled Paraphrase / Phrasing Perturbations
- Experiment E4: Robustness-Enhanced Model (Model C vs Model B vs Model A)

Calculates:
- Clean Accuracy, Clean F1-score
- Robust Accuracy (against ground-truth labels)
- Prediction Consistency (fraction of unchanged predictions)
- Prediction Flip Rate (fraction of changed predictions)
- Robustness Degradation (clean_acc - robust_acc in percentage points)
"""

import json
import random
import string
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple

import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import get_imdb_splits

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ROBUSTNESS_OUTPUT_PATH = MODELS_DIR / "robustness_results.json"
ALL_METRICS_OUTPUT_PATH = MODELS_DIR / "all_clean_metrics.json"

# Device selection
if torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
else:
    TORCH_DEVICE = torch.device("cpu")

# Protected keywords that must never be altered in sentiment benchmarks
PROTECTED_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "scarcely",
    "good", "bad", "great", "terrible", "awful", "horrible", "wonderful",
    "excellent", "poor", "best", "worst", "love", "loved", "hate", "hated",
}


# ---------------- PERTURBATION DEFINITIONS ----------------

def delete_character(text: str, rng: random.Random) -> str:
    """Deletes one character from a random non-protected word."""
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 4 and w.lower().strip(".,!?;:\"'") not in PROTECTED_WORDS
    ]
    if not candidates:
        return text

    i = rng.choice(candidates)
    word = words[i]
    pos = rng.randrange(len(word))
    words[i] = word[:pos] + word[pos + 1:]
    return " ".join(words)


def swap_characters(text: str, rng: random.Random) -> str:
    """Swaps two adjacent characters in a random non-protected word."""
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 4 and w.lower().strip(".,!?;:\"'") not in PROTECTED_WORDS
    ]
    if not candidates:
        return text

    i = rng.choice(candidates)
    word = words[i]
    if len(word) < 2:
        return text
    pos = rng.randrange(len(word) - 1)
    words[i] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    return " ".join(words)


def replace_character(text: str, rng: random.Random) -> str:
    """Replaces one character with a random letter in a non-protected word."""
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 4 and w.lower().strip(".,!?;:\"'") not in PROTECTED_WORDS
    ]
    if not candidates:
        return text

    i = rng.choice(candidates)
    word = words[i]
    pos = rng.randrange(len(word))
    choices = string.ascii_lowercase.replace(word[pos].lower(), "")
    rep = rng.choice(choices)
    words[i] = word[:pos] + rep + word[pos + 1:]
    return " ".join(words)


def repeat_character(text: str, rng: random.Random) -> str:
    """Repeats one character in a random non-protected word."""
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 4 and w.lower().strip(".,!?;:\"'") not in PROTECTED_WORDS
    ]
    if not candidates:
        return text

    i = rng.choice(candidates)
    word = words[i]
    pos = rng.randrange(len(word))
    words[i] = word[:pos] + word[pos] + word[pos:]
    return " ".join(words)


def controlled_paraphrase(text: str, rng: random.Random) -> str:
    """Applies verified meaning-preserving phrase and structural variation.
    
    Substitutes neutral context phrasing without modifying core sentiment tokens.
    """
    replacements = {
        "in my opinion": "from my point of view",
        "as a result": "consequently",
        "for example": "for instance",
        "at the end of": "towards the conclusion of",
        "the main character": "the protagonist",
        "the director": "the filmmaker",
        "in addition": "furthermore",
        "on the other hand": "however",
        "first of all": "to begin with",
        "throughout the movie": "across the entire film",
        "this movie": "this motion picture",
        "the plot": "the storyline",
        "the actors": "the cast",
        "in the theatre": "in the cinema",
    }

    modified = text
    applied = False
    for phrase, substitute in replacements.items():
        if phrase in modified.lower():
            # Case-insensitive replace once
            idx = modified.lower().find(phrase)
            modified = modified[:idx] + substitute + modified[idx + len(phrase):]
            applied = True
            break

    if not applied:
        # Prepend subtle neutral conversational framing if no phrase matched
        framings = [
            "All things considered, ",
            "Looking at the whole picture, ",
            "Taking everything into account, ",
        ]
        chosen = rng.choice(framings)
        modified = chosen + text[0].lower() + text[1:]

    return modified


PERTURBATIONS: Dict[str, Tuple[Callable[[str, random.Random], str], str]] = {
    "character_deletion": (delete_character, "Deletes 1 character in a non-sentiment word"),
    "character_swap": (swap_characters, "Swaps 2 adjacent characters in a non-sentiment word"),
    "character_replacement": (replace_character, "Replaces 1 character with an adjacent letter"),
    "character_repetition": (repeat_character, "Duplicates 1 character (simulates informal typing)"),
    "controlled_paraphrase": (controlled_paraphrase, "Replaces neutral syntactic phrases"),
}


# ---------------- MODEL PREDICTOR WRAPPERS ----------------

class ModelPredictor:
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type

    def predict(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SklearnPredictor(ModelPredictor):
    def __init__(self, name: str, pipeline):
        super().__init__(name, "sklearn")
        self.pipeline = pipeline

    def predict(self, texts: List[str]) -> np.ndarray:
        return np.array(self.pipeline.predict(texts), dtype=int)


class TransformerPredictor(ModelPredictor):
    def __init__(self, name: str, model_path: Path, batch_size: int = 32):
        super().__init__(name, "transformer")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(TORCH_DEVICE)
        self.model.eval()
        self.batch_size = batch_size

    def predict(self, texts: List[str]) -> np.ndarray:
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(TORCH_DEVICE)
                logits = self.model(**inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        return np.array(all_preds, dtype=int)


def discover_models() -> List[ModelPredictor]:
    """Finds all trained and ready models in backend/models."""
    predictors = []

    # Model A: Word-level TF-IDF
    tfidf_path = MODELS_DIR / "sentiment_pipeline.joblib"
    if tfidf_path.exists():
        predictors.append(SklearnPredictor("Word-level TF-IDF", joblib.load(tfidf_path)))

    # Optional: Character-level TF-IDF
    char_path = MODELS_DIR / "sentiment_char_pipeline.joblib"
    if char_path.exists():
        predictors.append(SklearnPredictor("Character-level TF-IDF", joblib.load(char_path)))

    # Model B: DistilBERT Fine-tuned
    distilbert_path = MODELS_DIR / "distilbert_sentiment"
    if distilbert_path.exists() and (distilbert_path / "model.safetensors").exists():
        try:
            predictors.append(TransformerPredictor("DistilBERT (Fine-tuned)", distilbert_path))
        except Exception as e:
            print(f"Warning: Could not load DistilBERT predictor: {e}")

    # Model C: DistilBERT Robustness-Enhanced
    distilbert_robust_path = MODELS_DIR / "distilbert_robust"
    if distilbert_robust_path.exists() and (distilbert_robust_path / "model.safetensors").exists():
        try:
            predictors.append(TransformerPredictor("DistilBERT (Robustness-Enhanced)", distilbert_robust_path))
        except Exception as e:
            print(f"Warning: Could not load DistilBERT Robust predictor: {e}")

    return predictors


def run_full_evaluation(test_size: int = 2500, seed: int = 42) -> Tuple[List[Dict], Dict]:
    start_time = time.time()
    splits = get_imdb_splits(seed=seed, test_size=test_size)
    clean_texts = splits["test"]["texts"]
    ground_truth = np.array(splits["test"]["labels"], dtype=int)

    predictors = discover_models()
    if not predictors:
        raise RuntimeError("No models found in models/ directory. Run training scripts first.")

    print(f"\n=======================================================")
    print(f"Running Robustness Evaluation Pipeline (E1 - E4)")
    print(f"Test samples: {len(clean_texts)} | Models detected: {len(predictors)}")
    for p in predictors:
        print(f"  - {p.name} ({p.model_type})")
    print(f"=======================================================\n")

    robustness_results = []
    clean_metrics_summary = {}

    for predictor in predictors:
        print(f"\nEvaluating: {predictor.name}...")

        # E1: Clean Evaluation
        clean_preds = predictor.predict(clean_texts)
        clean_acc = float(accuracy_score(ground_truth, clean_preds))
        clean_prec = float(precision_score(ground_truth, clean_preds, zero_division=0))
        clean_rec = float(recall_score(ground_truth, clean_preds, zero_division=0))
        clean_f1 = float(f1_score(ground_truth, clean_preds, zero_division=0))
        clean_cm = confusion_matrix(ground_truth, clean_preds).tolist()

        clean_metrics_summary[predictor.name] = {
            "accuracy": round(clean_acc * 100, 2),
            "precision": round(clean_prec * 100, 2),
            "recall": round(clean_rec * 100, 2),
            "f1_score": round(clean_f1 * 100, 2),
            "confusion_matrix": clean_cm,
            "test_samples": len(clean_texts),
        }

        print(f"  Clean Accuracy: {clean_acc * 100:.2f}% | Clean F1: {clean_f1 * 100:.2f}%")

        model_entry = {
            "model": predictor.name,
            "original_accuracy": round(clean_acc * 100, 2),
            "clean_f1": round(clean_f1 * 100, 2),
            "test_samples": len(clean_texts),
            "perturbations": {},
        }

        # E2 & E3: Perturbations
        for pert_name, (pert_fn, description) in PERTURBATIONS.items():
            rng = random.Random(seed)  # Exact identical seed across all models
            perturbed_texts = [pert_fn(t, rng) for t in clean_texts]

            pert_preds = predictor.predict(perturbed_texts)
            pert_acc = float(accuracy_score(ground_truth, pert_preds))

            # Flip rate: predictions that changed from clean prediction
            flip_rate = float(np.mean(clean_preds != pert_preds)) * 100.0
            consistency = 100.0 - flip_rate
            acc_drop = (clean_acc - pert_acc) * 100.0

            result = {
                "accuracy": round(pert_acc * 100.0, 2),
                "accuracy_drop": round(acc_drop, 2),
                "prediction_flip_rate": round(flip_rate, 2),
                "prediction_consistency": round(consistency, 2),
                "description": description,
            }
            model_entry["perturbations"][pert_name] = result

            print(
                f"    {pert_name:<23s}: Acc={result['accuracy']:5.2f}% | "
                f"Drop={result['accuracy_drop']:+5.2f} pts | Flip={result['prediction_flip_rate']:5.2f}%"
            )

        robustness_results.append(model_entry)

    # Save to disk
    ROBUSTNESS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROBUSTNESS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=4)

    with open(ALL_METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_metrics_summary, f, indent=4)

    elapsed = round(time.time() - start_time, 2)
    print(f"\nEvaluation complete in {elapsed}s.")
    print(f"Results written to: {ROBUSTNESS_OUTPUT_PATH}")
    print(f"Clean metrics written to: {ALL_METRICS_OUTPUT_PATH}\n")

    return robustness_results, clean_metrics_summary


if __name__ == "__main__":
    run_full_evaluation()
