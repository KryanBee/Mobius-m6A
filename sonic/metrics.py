"""Score validation and training-only balanced class weights."""

import numpy as np


def validate_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite positive-class probabilities in [0,1]")
    return scores


def class_weights(labels, mode):
    """Equal total contribution per class, mean training-site weight = 1.

    Use once on training labels, not per minibatch or on validation labels.
    Normalising keeps the overall loss scale comparable when weighting is off.
    """
    labels = np.asarray(labels)
    if set(labels) != {0, 1}:
        raise ValueError("Training needs both binary classes")
    counts = np.bincount(labels.astype(int), minlength=2)
    if mode not in ("balanced", "none"):
        raise ValueError("Unsupported class weight mode")
    weights = len(labels) / (2 * counts) if mode == "balanced" else np.ones(2)
    return {"mode": mode, "counts": counts.tolist(), "weights": weights.tolist(),
            "formula": "N / (2 * N_class)" if mode == "balanced" else "1"}
