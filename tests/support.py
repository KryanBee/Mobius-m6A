"""Small deterministic signal fixtures, independent of research data or labels."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
CSV_COLUMNS = ["transcript_id", "transcript_position", "score"]


def signal_records(count=6, *, seed=742, minimum_reads=20):
    """Return varied finite nine-channel bags with deliberately unsorted IDs."""
    rng = np.random.default_rng(seed)
    sequences = ("AAGACCA", "CTGACTG", "GGGACTA")
    records = []
    for i in range(count):
        reads = rng.lognormal(mean=0.2 * (i % 2), sigma=0.6,
                              size=(minimum_reads + i % 4, 9))
        transcript = f"unannotated_transcript_{count - i}"
        position = str(3 * (count - i))
        records.append({transcript: {position: {sequences[i % 3]: reads.tolist()}}})
    return records


def write_signals(path, records):
    path = Path(path)
    text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in records)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def site_keys(records):
    return [(transcript, int(position)) for record in records
            for transcript, positions in record.items() for position in positions]


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise AssertionError(f"Incorrect prediction columns: {reader.fieldnames}")
        rows = list(reader)
    keys = [(row["transcript_id"], int(row["transcript_position"])) for row in rows]
    scores = np.asarray([float(row["score"]) for row in rows])
    if len(keys) != len(set(keys)) or not np.isfinite(scores).all():
        raise AssertionError("Duplicate keys or nonfinite prediction scores")
    if not ((0 <= scores) & (scores <= 1)).all():
        raise AssertionError("Prediction score outside [0, 1]")
    return keys, scores


def write_targets(path, records, values=None, *, reverse=False, gene_metadata=False):
    """Generate targets only inside a test's temporary directory."""
    keys = site_keys(records)
    values = [0.0, 0.25, 0.0, 0.8, 0.0, 1.0] if values is None else values
    if len(keys) != len(values):
        raise ValueError("A generated target is required for each generated site")
    rows = [dict(transcript_id=key[0], transcript_position=key[1], label=value)
            for key, value in zip(keys, values)]
    if gene_metadata:
        for row in rows:
            row["gene_id"] = "optional_test_metadata"
    if reverse:
        rows.reverse()
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
