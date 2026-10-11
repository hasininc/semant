"""Fine-tuning pipeline for Model B: DistilBERT Sentiment Classifier.

Designed for Apple Silicon M1 (8GB RAM) with MPS acceleration, modest batch size,
and strict held-out test set evaluation.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import get_imdb_splits


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = float(accuracy_score(all_labels, all_preds))
    prec = float(precision_score(all_labels, all_preds, zero_division=0))
    rec = float(recall_score(all_labels, all_preds, zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


def train_distilbert(
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 2500,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str = "models/distilbert_sentiment",
    metrics_path: str = "models/distilbert_metrics.json",
):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection: MPS (Apple Silicon) -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"Training Model B: DistilBERT Sentiment Classifier")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Train samples: {train_size} | Val samples: {val_size} | Test samples: {test_size}")
    print(f"=======================================================\n")

    # Load deterministic splits
    splits = get_imdb_splits(
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    ).to(device)

    train_dataset = TextClassificationDataset(
        splits["train"]["texts"],
        splits["train"]["labels"],
        tokenizer,
        max_length=max_length,
    )
    val_dataset = TextClassificationDataset(
        splits["val"]["texts"],
        splits["val"]["labels"],
        tokenizer,
        max_length=max_length,
    )
    test_dataset = TextClassificationDataset(
        splits["test"]["texts"],
        splits["test"]["labels"],
        tokenizer,
        max_length=max_length,
    )

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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] "
                    f"Loss: {total_loss / step:.4f}"
                )

        # Validation after each epoch
        val_metrics = evaluate(model, val_loader, device)
        print(f"\n--- Epoch {epoch} Validation: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1_score']:.4f} ---")

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            print("=> New best validation checkpoint. Saving...")
            model.save_pretrained(best_weights_path)
            tokenizer.save_pretrained(best_weights_path)

    # Final evaluation on the held-out test split using best model
    print("\nEvaluating best checkpoint on held-out test set (2,500 samples)...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_weights_path).to(device)
    test_metrics = evaluate(best_model, test_loader, device)

    elapsed_time = round(time.time() - start_time, 2)

    final_metrics = {
        "model": "DistilBERT (Fine-tuned)",
        "base_model": base_model_name,
        "dataset": "IMDb",
        "training_samples": train_size,
        "validation_samples": val_size,
        "testing_samples": test_size,
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

    print("\n================== Model B Results ==================")
    for k, v in final_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"Confusion Matrix:\n  {final_metrics['confusion_matrix']}")
    print(f"\nModel saved to: {best_weights_path}")
    print(f"Metrics saved to: {metrics_file}")
    print("=====================================================\n")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model B: DistilBERT Sentiment Classifier")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_distilbert(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        seed=args.seed,
    )
