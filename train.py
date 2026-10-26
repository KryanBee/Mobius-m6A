#!/usr/bin/env python3
"""Fit Mobius using externally supplied RNA signal data and site labels."""

import argparse
from pathlib import Path
import zlib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", nargs=3, action="append", required=True,
        metavar=("SOURCE_ID", "SIGNAL_JSON", "LABELS"),
        help="repeat for each source; labels are CSV/TSV/data.info with transcript_id, transcript_position, label",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="new output directory; its model/ subdirectory is usable by predict.py --model",
    )
    parser.add_argument("--temp-dir", type=Path, help="existing scratch directory for temporary read caches")
    args = parser.parse_args(argv)
    try:
        from sonic.fitting import train_model
        result = train_model(args.dataset, args.output, temp_dir=args.temp_dir)
    except (ValueError, OSError, EOFError, KeyError, TypeError, RuntimeError, zlib.error) as error:
        parser.exit(2, f"Training failed: {error}\n")
    print(f"Saved trained model: {result / 'model'}")


if __name__ == "__main__":
    main()
