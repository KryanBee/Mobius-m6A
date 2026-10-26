"""Validate numeric model inputs; never silently impute undefined features."""

import numpy as np


def transform(values, scaler):
    values = np.asarray(values, dtype=np.float64)
    if scaler is not None:
        values = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite numeric input after scaling")
    return values
