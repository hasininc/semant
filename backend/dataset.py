"""Centralized dataset management for SemanticShield.

Ensures strict separation between train, validation, and held-out test splits
to prevent data leakage across all models (TF-IDF, DistilBERT, DistilBERT-Robust).
"""

from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset


def get_imdb_splits(
    seed: int = 42,
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 2500,
) -> Dict[str, Dict[str, List]]:
    """Loads Stanford IMDb dataset with strictly disjoint splits.

    - Train and Validation are drawn from official dataset['train'] (25,000 samples).
    - Held-out Test is drawn from official dataset['test'] (25,000 samples).
    - Train and Validation indices are strictly non-overlapping.
    """
    raw_dataset = load_dataset("stanfordnlp/imdb")

    # Shuffled training pool
    shuffled_train = raw_dataset["train"].shuffle(seed=seed)

    if train_size + val_size > len(shuffled_train):
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) exceeds "
            f"available training samples ({len(shuffled_train)})."
        )

    train_data = shuffled_train.select(range(0, train_size))
    val_data = shuffled_train.select(range(train_size, train_size + val_size))

    # Shuffled test pool - STRICTLY held out from dataset['test']
    shuffled_test = raw_dataset["test"].shuffle(seed=seed)
    test_data = shuffled_test.select(range(0, test_size))

    # Sanity checks
    assert len(train_data) == train_size
    assert len(val_data) == val_size
    assert len(test_data) == test_size

    # Verify label distributions
    train_labels = list(train_data["label"])
    val_labels = list(val_data["label"])
    test_labels = list(test_data["label"])

    return {
        "train": {
            "texts": list(train_data["text"]),
            "labels": train_labels,
        },
        "val": {
            "texts": list(val_data["text"]),
            "labels": val_labels,
        },
        "test": {
            "texts": list(test_data["text"]),
            "labels": test_labels,
        },
    }


if __name__ == "__main__":
    splits = get_imdb_splits()
    print("Split summary:")
    for split_name, data in splits.items():
        pos_count = sum(data["labels"])
        neg_count = len(data["labels"]) - pos_count
        print(
            f"  {split_name:5s}: {len(data['texts']):5d} samples "
            f"(pos: {pos_count}, neg: {neg_count})"
        )
