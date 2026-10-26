"""Check the bundled example's model outputs, without ground-truth labels."""
import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV generated from examples/test.json")
    args = parser.parse_args()
    try:
        tables = []
        for path in (Path(__file__).with_name("expected_predictions.csv"), args.predictions):
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["transcript_id", "transcript_position", "score"]:
                    raise ValueError(f"Unexpected CSV columns in {path}")
                rows = list(reader)
                if any(None in row or None in row.values() for row in rows):
                    raise ValueError(f"Each CSV row must have exactly three values in {path}")
                tables.append(rows)
        expected, actual = tables
        if len(actual) != len(expected):
            raise ValueError(f"Expected {len(expected)} rows; found {len(actual)}")
        for number, (reference, result) in enumerate(zip(expected, actual), 1):
            if any(reference[key] != result[key] for key in ("transcript_id", "transcript_position")):
                raise ValueError(f"Site identity/order differs at row {number}")
            score = float(result["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Score must be finite and in [0, 1] at row {number}")
            if not math.isclose(score, float(reference["score"]), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Frozen-model score differs at row {number}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(f"PASS: three-column CSV, {len(expected)} sites, finite scores in [0, 1]; "
          "site identifiers/order and expected scores match (absolute tolerance 1e-8).")


if __name__ == "__main__":
    main()
