"""Public prediction contract, frozen numerical parity and safe failure modes."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.support import (MODEL, ROOT, read_predictions, signal_records,
                           site_keys, write_signals)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sonic release tests ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def predict(self, source, target="scores.csv", **kwargs):
        from sonic.prediction import predict_file
        output = self.directory / target
        count = predict_file(source, output, **kwargs)
        keys, scores = read_predictions(output)
        self.assertEqual(count, len(keys))
        return keys, scores

    def test_bundled_example_matches_frozen_expected_scores(self):
        from sonic.data import raw_bags
        example = ROOT / "examples/test.json"
        self.assertLess(example.stat().st_size, 1_000_000)
        parsed = list(raw_bags(example, "example"))
        self.assertGreater(len(parsed), 0)
        self.assertTrue(all(len(reads) >= 20 for _, reads in parsed))
        expected_keys, expected_scores = read_predictions(ROOT / "examples/expected_predictions.csv")
        # A small chunk forces a real boundary within the bundled example.
        actual_keys, actual_scores = self.predict(example, chunk_sites=2)
        self.assertEqual(actual_keys, expected_keys)
        np.testing.assert_allclose(actual_scores, expected_scores, rtol=0, atol=1e-8)

    def test_unknown_identifiers_input_order_and_chunk_sizes(self):
        records = signal_records()
        source = write_signals(self.directory / "unknown.json", records)
        keys, scores = self.predict(source, "single.csv", chunk_sites=1)
        whole_keys, whole_scores = self.predict(source, "together.csv", chunk_sites=64)
        self.assertEqual(keys, site_keys(records))
        self.assertEqual(keys, whole_keys)
        np.testing.assert_array_equal(scores, whole_scores)

    def test_cli_gzip_spaces_and_unrelated_working_directory(self):
        records = signal_records(3)
        source = write_signals(self.directory / "input with spaces.json.gz", records)
        output = self.directory / "output with spaces.csv"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(ROOT / "predict.py"), "--input", str(source),
             "--output", str(output), "--chunk-sites", "2"],
            cwd=self.directory, env=environment, text=True, capture_output=True,
            timeout=90, check=False)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        keys, scores = read_predictions(output)
        self.assertEqual(keys, site_keys(records))
        plain = write_signals(self.directory / "same.json", records)
        _, expected = self.predict(plain)
        np.testing.assert_array_equal(scores, expected)

    def test_no_overwrite_and_input_output_alias_protection(self):
        from sonic.prediction import predict_file
        source = write_signals(self.directory / "input.json", signal_records(1))
        output = self.directory / "existing.csv"
        output.write_text("keep this existing file\n", encoding="utf-8")
        for target in (source, output):
            before = target.read_bytes()
            with self.subTest(target=target.name), self.assertRaises((ValueError, FileExistsError)):
                predict_file(source, target)
            self.assertEqual(target.read_bytes(), before)

    def test_late_invalid_site_leaves_no_partial_output_or_temporary_file(self):
        from sonic.prediction import predict_file
        records = signal_records(2)
        transcript = next(iter(records[-1]))
        position = next(iter(records[-1][transcript]))
        sequence = next(iter(records[-1][transcript][position]))
        records[-1][transcript][position][sequence] = records[-1][transcript][position][sequence][:19]
        source = write_signals(self.directory / "late-invalid.json", records)
        before = set(self.directory.iterdir())
        output = self.directory / "must-not-exist.csv"
        with self.assertRaisesRegex(ValueError, "20"):
            predict_file(source, output, chunk_sites=1)
        self.assertFalse(output.exists())
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_degenerate_signals_fail_without_zero_filling(self):
        from sonic.prediction import predict_file
        records = [{"unknown": {"10": {"AAGACCA": [[1.0] * 9 for _ in range(20)]}}}]
        source = write_signals(self.directory / "constant.json", records)
        output = self.directory / "constant.csv"
        with self.assertRaises(ValueError):
            predict_file(source, output)
        self.assertFalse(output.exists())

    def test_empty_input_fails_without_publishing_a_header_only_csv(self):
        from sonic.prediction import predict_file
        source = self.directory / "empty.json"
        source.write_text("", encoding="utf-8")
        output = self.directory / "empty.csv"
        with self.assertRaisesRegex(ValueError, "no sites|empty"):
            predict_file(source, output)
        self.assertFalse(output.exists())

    def test_corrupt_compressed_input_fails_without_traceback_or_partial_output(self):
        records = signal_records(3)
        # A valid first gzip member allows one site to be written to the staged
        # CSV before the second member fails with an invalid DEFLATE block type.
        first = gzip.compress("".join(json.dumps(row) + "\n" for row in records[:2]).encode())
        corrupt = bytearray(gzip.compress((json.dumps(records[2]) + "\n").encode()))
        corrupt[10] = 0xff
        source = self.directory / "corrupt.json.gz"
        source.write_bytes(first + corrupt)
        output = self.directory / "must-not-exist.csv"
        result = subprocess.run(
            [sys.executable, str(ROOT / "predict.py"), "--input", str(source),
             "--output", str(output), "--chunk-sites", "1"],
            cwd=self.directory, capture_output=True, text=True, timeout=90, check=False)
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertIn("Prediction failed:", result.stderr)
        self.assertIn("decompress", result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.directory.glob(".prediction-*.partial")), [])

    def test_corrupted_checkpoint_is_rejected_before_publishing(self):
        from sonic.prediction import predict_file
        source = write_signals(self.directory / "input.json", signal_records(1))
        for relative in ("trees.ubj", "encoder/weights.pt", "motif_reference.json"):
            copied = self.directory / relative.replace("/", "-")
            shutil.copytree(MODEL, copied)
            with (copied / relative).open("ab") as handle:
                handle.write(b"tampered")
            output = self.directory / (copied.name + ".csv")
            with self.subTest(file=relative), self.assertRaisesRegex(ValueError, "checksum|integrity|hash"):
                predict_file(source, output, model_path=copied)
            self.assertFalse(output.exists())

    def test_inference_does_not_fit_a_normalization_reference(self):
        source = write_signals(self.directory / "input.json", signal_records(2))
        with patch("sonic.rich_features.fit_reference",
                   side_effect=AssertionError("Inference must reuse fitted reference")):
            keys, _ = self.predict(source)
        self.assertEqual(len(keys), 2)


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sonic-input-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def parse_text(self, text):
        from sonic.data import raw_bags
        path = self.directory / "input.json"
        path.write_text(text, encoding="utf-8")
        return list(raw_bags(path, "fixture"))

    def test_read_minimum_is_inclusive_and_no_gene_lookup_needed(self):
        row = signal_records(1)[0]
        parsed = self.parse_text(json.dumps(row) + "\n")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0][1].shape, (20, 9))
        self.assertEqual(parsed[0][0]["transcript_id"], "unannotated_transcript_1")
        short = deepcopy(row)
        short["unannotated_transcript_1"]["3"]["AAGACCA"].pop()
        with self.assertRaisesRegex(ValueError, "20"):
            self.parse_text(json.dumps(short) + "\n")

    def test_duplicate_sites_and_duplicate_json_members_are_rejected(self):
        line = json.dumps(signal_records(1)[0])
        invalid = [line + "\n" + line + "\n", '{"same": {}, "same": {}}\n',
                   '{"tx": {"1": {}, "1": {}}}\n']
        for text in invalid:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                self.parse_text(text)

    def test_empty_malformed_and_invalid_identifiers_are_rejected(self):
        values = ["\n", "not-json\n", "[]\n", "{}\n"]
        reads = signal_records(1)[0]["unannotated_transcript_1"]["3"]["AAGACCA"]
        for transcript, position, sequence in (
                ("", "3", "AAGACCA"), (" padded ", "3", "AAGACCA"),
                ("tx", "-1", "AAGACCA"), ("tx", "1.5", "AAGACCA"),
                ("tx", "01", "AAGACCA"), ("tx", str(2 ** 63), "AAGACCA"),
                ("tx", "3", "AAGACC"), ("tx", "3", "AAGACCN")):
            values.append(json.dumps({transcript: {position: {sequence: reads}}}))
        for text in values:
            with self.subTest(text=text[:55]), self.assertRaises(ValueError):
                self.parse_text(text)

    def test_wrong_channel_count_nonnumeric_and_nonfinite_reads_are_rejected(self):
        original = signal_records(1)[0]["unannotated_transcript_1"]["3"]["AAGACCA"]
        arrays = [[], [[1.0] * 8 for _ in range(20)]]
        for bad in ("1.0", True, None, float("nan"), float("inf")):
            reads = deepcopy(original)
            reads[0][0] = bad
            arrays.append(reads)
        for reads in arrays:
            text = json.dumps({"tx": {"3": {"AAGACCA": reads}}})
            with self.subTest(value=str(reads)[:40]), self.assertRaises(ValueError):
                self.parse_text(text)


if __name__ == "__main__":
    unittest.main()
