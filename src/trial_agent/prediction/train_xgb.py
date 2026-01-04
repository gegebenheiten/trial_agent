"""
Placeholder for a lightweight gradient boosting or logistic regression training script.
For the MVP we keep heuristics; extend this file to train on structured features
and persisted labels once you have >100 labeled trials.
"""

from pathlib import Path
from typing import List


def load_features_and_labels(processed_path: Path, labels_path: Path) -> List:
    """
    Wire in pandas or polars here when you are ready to train a real model.
    """
    raise NotImplementedError("Training pipeline not implemented in the MVP.")


if __name__ == "__main__":
    print("Training pipeline is intentionally left as a stub for now.")

