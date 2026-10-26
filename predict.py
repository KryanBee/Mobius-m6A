#!/usr/bin/env python3
"""Predict m6A site scores from RNA signal data using the saved Mobius model."""
import argparse
from pathlib import Path
import sys
import time
import zlib


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input", required=True, type=Path,
                         help="JSON or JSON.gz, one site per line, at least 20 reads per site")
    command.add_argument("--output", required=True, type=Path,
                         help="New three-column CSV path; existing files are never overwritten")
    command.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "model",
                         help="Model directory (default: model/ beside this script)")
    command.add_argument("--chunk-sites", type=int, default=2048,
                         help="Maximum sites processed together (default: 2048)")
    command.add_argument("--chunk-reads", type=int, default=200000,
                         help="Read budget per chunk; a complete site may exceed it (default: 200000)")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    started = time.perf_counter()
    try:
        from sonic.prediction import predict_file
        count = predict_file(args.input, args.output, args.model,
                             chunk_sites=args.chunk_sites, chunk_reads=args.chunk_reads)
    except (ValueError, OSError, EOFError, KeyError, TypeError, RuntimeError, zlib.error) as error:
        command.exit(2, f"Prediction failed: {error}\n")
    print(f"Wrote {count} site scores to {args.output} in {time.perf_counter() - started:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
