"""Fitting-only motif-relative read inputs, separate from immutable bag metadata.

motif_z_v1 replaces the nine globally scaled channels with nine five-mer z-scores.
It reuses the frozen tabular reference estimator: read-weighted population moments
pooled over all occurrences of each actual five-mer, without consulting labels.
"""

import numpy as np

from sonic.common import read_json, write_json
from sonic.features import READ_FEATURES, encode_sequence
from sonic.rich_features import fit_reference, validate_reference, validate_rich_reads

VERSION = "motif_z_v1"
REFERENCE_FILE = "read_motif_reference.json"
SCALING_UNIT = "fitting_read_motif_windows"


def fit_and_save(data, fitting, model_dir):
    reference = fit_reference(data, fitting)
    validate_reference(reference)
    write_json(model_dir / REFERENCE_FILE, {
        "schema_version": 1, "read_normalization": VERSION,
        "numeric_features": list(READ_FEATURES), "reference": reference})
    return reference


def load_reference(model_dir, bundle):
    """Validate the read branch contract; load_bundle checks its checksum first."""
    meta = bundle["preprocessing"]
    joint = bundle.get("schema_version") == 4 and bundle["model"] == "joint_network"
    if (not joint and (bundle.get("schema_version") != 3 or bundle["model"] != "bag_network")
            or bundle.get("read_normalization") != VERSION
            or bundle.get("feature_version") != "reads9_v1"
            or bundle["numeric_features"] != list(READ_FEATURES)
            or meta.get("scaler", "missing") is not None
            or meta.get("scaling_unit") != SCALING_UNIT
            or meta.get("motif_reference_file") != REFERENCE_FILE
            or REFERENCE_FILE not in bundle["files"] or (not joint and "feature_transform" in bundle)):
        raise ValueError("Invalid motif-normalized neural bundle")
    saved = read_json(model_dir / REFERENCE_FILE)
    if (saved.get("schema_version") != 1 or saved.get("read_normalization") != VERSION
            or saved.get("numeric_features") != list(READ_FEATURES)):
        raise ValueError("Invalid saved neural motif-reference schema")
    reference = saved["reference"]
    validate_reference(reference)
    for key in ("fitting_sites", "fitting_reads"):
        if type(reference.get(key)) is not int or reference[key] <= 0:
            raise ValueError(f"Invalid motif-reference {key}")
    digest = reference.get("fitting_keys_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError("Invalid motif-reference fitting-key digest")
    return reference


class MotifReadScaler:
    """Small fold-local lookup, bound to selected rows of one FeatureData object.

    Only sequence metadata is indexed here; no read values or statistics are
    fitted. Never store this object or its learned moments in FeatureData's cache.
    Unselected rows (including reserved test) are neither normalized nor required
    to have reference motifs. Unseen selected motifs fail before model execution.
    """

    def __init__(self, data, rows, reference):
        validate_reference(reference)
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
                or (rows < 0).any() or (rows >= len(data.sites)).any()):
            raise ValueError("Invalid rows for motif read normalization")
        sequences, inverse = np.unique(data.sites.iloc[rows].sequence_7mer.to_numpy(), return_inverse=True)
        encode_sequence(sequences)
        means, scales = [], []
        for sequence in sequences:
            motifs = [sequence[w:w+5] for w in range(3)]
            missing = sorted(set(motifs) - reference["motifs"].keys())
            if missing:
                raise ValueError(f"Unseen motif(s) {missing}; {VERSION} has no fallback")
            means.append([reference["motifs"][motif]["mean"] for motif in motifs])
            scales.append([reference["motifs"][motif]["sd"] for motif in motifs])
        self.data = data
        self.codes = np.full(len(data.sites), -1, dtype=np.int32)
        self.codes[rows] = inverse
        self.mean = np.asarray(means, dtype=np.float64).reshape(-1, 9)
        self.scale = np.asarray(scales, dtype=np.float64).reshape(-1, 9)
        for value in (self.codes, self.mean, self.scale):
            value.setflags(write=False)

    def transform(self, data, rows, counts, values):
        rows = np.asarray(rows)
        if (data is not self.data or rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
                or (rows < 0).any() or (rows >= len(self.codes)).any()):
            raise ValueError("Motif scaler belongs to different data or invalid rows")
        codes = self.codes[rows]
        if (codes < 0).any():
            raise ValueError("Rows were not selected for this motif scaler")
        values = validate_rich_reads(values)
        numeric = ((values - np.repeat(self.mean[codes], counts, axis=0))
                   / np.repeat(self.scale[codes], counts, axis=0))
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite motif-normalized read values")
        return numeric
