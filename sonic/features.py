"""Deterministic feature definitions: change the version when semantics change."""

import numpy as np

READ_FEATURES = [f"{measure}_{window}"
                 for window in ("minus1", "center", "plus1")
                 for measure in ("dwell_time", "signal_sd", "signal_mean")]

SEQUENCE_ENCODING = "base_onehot_v1"
SEQUENCE_FEATURES = [f"base_{position}_{base}" for position in range(7) for base in "ACGT"]


def mean9(reads):
    return reads.mean(axis=0, dtype=np.float64)


# Add a new recipe here when EDA supplies richer summaries. Keep old recipes so
# saved models still extract precisely the features used during their training.
SUMMARY_RECIPES = {"mean9_v1": ([f"mean_{x}" for x in READ_FEATURES], mean9)}


def summary_recipe(version):
    if version not in SUMMARY_RECIPES:
        raise ValueError(f"Unknown summary recipe: {version}")
    return SUMMARY_RECIPES[version]


def encode_sequence(sequences):
    """Seven one-hot bases preserve all three overlapping five-base windows."""
    sequences = list(sequences)
    alphabet = {b: i for i, b in enumerate("ACGT")}
    if any(not isinstance(s, str) or len(s) != 7 or set(s) - alphabet.keys() for s in sequences):
        raise ValueError("sequence_7mer must contain exactly seven uppercase A/C/G/T bases")
    codes = np.array([[alphabet[b] for b in s] for s in sequences], dtype=np.int64)
    return np.eye(4, dtype=np.float32)[codes].reshape(len(sequences), 28)


def validate_reads(reads):
    reads = np.asarray(reads, dtype=np.float64)
    if reads.ndim != 2 or reads.shape[1] != 9 or len(reads) == 0:
        raise ValueError("Every site must have a nonempty n-by-9 read bag; no sites are dropped")
    if not np.isfinite(reads).all():
        raise ValueError("Nonfinite read measurements; no imputation policy is defined")
    return reads
