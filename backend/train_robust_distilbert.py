"""Training pipeline for Model C: Robustness-Enhanced DistilBERT.

Applies label-preserving character-level noise augmentation exclusively to
the training split (protecting sentiment keywords and negation words),
and evaluates on the exact same held-out test set as Model A and Model B.
"""

import argparse
import json
import random
import string
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from dataset import get_imdb_splits
from train_distilbert import TextClassificationDataset, evaluate

# Reserved keywords to preserve sentiment polarity during augmentation
PROTECTED_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "scarcely",
    "good", "bad", "great", "terrible", "awful", "horrible", "wonderful",
    "excellent", "poor", "best", "worst", "love", "loved", "hate", "hated",
    "like", "dislike", "enjoy", "enjoyed", "boring", "brilliant", "trash",
    "superb", "dreadful", "masterpiece", "waste", "fantastic", "flawless",
}


def augment_text(text: str, rng: random.Random) -> str:
    """Applies a single, label-preserving character perturbation to a non-protected word."""
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 4 and w.lower().strip(".,!?;:\"'") not in PROTECTED_WORDS
    ]

    if not candidates:
        return text

    idx = rng.choice(candidates)
    word = words[idx]

    # Clean punctuation boundary
    prefix = ""
    suffix = ""
    while word and word[0] in string.punctuation:
        prefix += word[0]
        word = word[1:]
    while word and word[-1] in string.punctuation:
        suffix = word[-1] + suffix
        word = word[:-1]

    if len(word) < 4:
        return text

    op = rng.choice(["swap", "delete", "replace", "repeat"])

    if op == "swap":
        pos = rng.randrange(len(word) - 1)
        mod = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    elif op == "delete":
        pos = rng.randrange(len(word))
        mod = word[:pos] + word[pos + 1:]
    elif op == "replace":
        pos = rng.randrange(len(word))
        char_choices = string.ascii_lowercase.replace(word[pos].lower(), "")
        rep = rng.choice(char_choices)
        mod = word[:pos] + rep + word[pos + 1:]
    else:  # repeat
        pos = rng.randrange(len(word))
        mod = word[:pos] + word[pos] + word[pos:]

    words[idx] = prefix + mod + suffix
    return " ".join(words)


def create_augmented_training_data(
    texts: List[str],
    labels: List[int],
    augmentation_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Generates augmented examples exclusively from training data."""
    rng = random.Random(seed)
    augmented_texts = list(texts)
    augmented_labels = list(labels)

    num_to_augment = int(len(texts) * augmentation_ratio)
    sample_indices = rng.sample(range(len(texts)), num_to_augment)

    for i in sample_indices:
        perturbed = augment_text(texts[i], rng)
        augmented_texts.append(perturbed)
        augmented_labels.append(labels[i])  # Label preserved

    meta = {
        "original_train_count": len(texts),
        "augmented_samples_added": num_to_augment,
        "total_augmented_train_count": len(augmented_texts),
        "augmentation_ratio": augmentation_ratio,
        "protected_keywords_count": len(PROTECTED_WORDS),
        "seed": seed,
    }
    return augmented_texts, augmented_labels, meta


def train_robust_distilbert(
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 2500,
    augmentation_ratio: float = 0.5,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str = "models/distilbert_robust",
    metrics_path: str = "models/distilbert_robust_metrics.json",
):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"Training Model C: Robustness-Enhanced DistilBERT")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Augmentation Ratio: {augmentation_ratio} on training split only")
    print(f"=======================================================\n")

    splits = get_imdb_splits(
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # Augment training data strictly from splits['train']
    aug_texts, aug_labels, aug_meta = create_augmented_training_data(
        splits["train"]["texts"],
        splits["train"]["labels"],
        augmentation_ratio=augmentation_ratio,
        seed=seed,
    )
    print(f"Original train size: {len(splits['train']['texts'])}")
    print(f"Augmented train size: {len(aug_texts)} (+{aug_meta['augmented_samples_added']} augmented)")

    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    ).to(device)

    train_dataset = TextClassificationDataset(aug_texts, aug_labels, tokenizer, max_length=max_length)
    val_dataset = TextClassificationDataset(splits["val"]["texts"], splits["val"]["labels"], tokenizer, max_length=max_length)
    test_dataset = TextClassificationDataset(splits["test"]["texts"], splits["test"]["labels"], tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_weights_path = Path(output_dir)
    best_weights_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {total_loss / step:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"\n--- Epoch {epoch} Validation: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1_score']:.4f} ---")

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            print("=> New best validation checkpoint. Saving...")
            model.save_pretrained(best_weights_path)
            tokenizer.save_pretrained(best_weights_path)

    print("\nEvaluating Model C on held-out test set (2,500 samples)...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_weights_path).to(device)
    test_metrics = evaluate(best_model, test_loader, device)

    elapsed_time = round(time.time() - start_time, 2)

    final_metrics = {
        "model": "DistilBERT (Robustness-Enhanced)",
        "base_model": base_model_name,
        "dataset": "IMDb",
        "training_samples": len(aug_texts),
        "clean_training_samples": train_size,
        "augmented_training_samples": aug_meta["augmented_samples_added"],
        "validation_samples": val_size,
        "testing_samples": test_size,
        "augmentation_metadata": aug_meta,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": lr,
        "training_duration_seconds": elapsed_time,
        "device": str(device),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "confusion_matrix": test_metrics["confusion_matrix"],
    }

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    print("\n================== Model C Results ==================")
    for k, v in final_metrics.items():
        if k not in ("confusion_matrix", "augmentation_metadata"):
            print(f"  {k}: {v}")
    print(f"Confusion Matrix:\n  {final_metrics['confusion_matrix']}")
    print(f"\nModel saved to: {best_weights_path}")
    print(f"Metrics saved to: {metrics_file}")
    print("=====================================================\n")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model C: Robustness-Enhanced DistilBERT")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--aug-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_robust_distilbert(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        augmentation_ratio=args.aug_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        seed=args.seed,
    )
