"""A bounded fixed-recipe fit and exact externally supplied target contracts."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.support import (MODEL, ROOT, read_predictions, signal_records,
                           site_keys, write_signals, write_targets)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecipeTests(unittest.TestCase):
    def test_fixed_recipe_rejects_stale_weights_lengths_and_settings(self):
        from copy import deepcopy
        from sonic.fitting import load_recipe
        selected = load_recipe()
        changes = [
            ("encoder", ("fixed_steps",), 12),
            ("component", ("fixed_steps",), 206),
            ("encoder", ("config", "class_weight"), "balanced"),
            ("component", ("config", "class_weight"), "balanced"),
            ("component", ("config", "params", "n_estimators"), 206),
            ("component", ("config", "params", "max_depth"), 7),
            ("encoder", ("config", "params", "learning_rate"), 0.01),
        ]
        for component, path, value in changes:
            changed = deepcopy(selected)
            target = changed[component]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(component=component, path=path), \
                    patch("sonic.fitting.read_json", return_value=changed), \
                    self.assertRaisesRegex(ValueError, "selected final recipe|Selected hybrid recipe"):
                load_recipe()


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sonic-label-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = signal_records()

    def test_fractional_labels_use_strictly_positive_rule_and_ignore_optional_genes(self):
        from sonic.fitting import read_labels
        values = [0, 1e-12, 0.25, 0.8, 1, 0]
        target = write_targets(self.directory / "targets.csv", self.records, values,
                               gene_metadata=True)
        actual = read_labels(target, "synthetic")
        self.assertEqual(actual.label.tolist(), [0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(actual.label_original, values)
        self.assertTrue(actual.source_id.eq("synthetic").all())

    def test_nonfinite_and_out_of_range_targets_are_rejected(self):
        from sonic.fitting import read_labels
        for bad in (-0.1, 1.1, float("nan"), float("inf"), "invalid"):
            target = write_targets(self.directory / "invalid.csv", self.records,
                                   [bad, 0, 0, 1, 0.25, 0])
            with self.subTest(value=bad), self.assertRaises(ValueError):
                read_labels(target, "fixture")

    def test_course_info_filename_and_compressed_targets(self):
        from sonic.fitting import read_labels
        target = write_targets(self.directory / "data.info", self.records)
        expected = read_labels(target, "source")
        compressed = self.directory / "data.info.gz"
        with gzip.open(compressed, "wb") as handle:
            handle.write(target.read_bytes())
        pd.testing.assert_frame_equal(read_labels(compressed, "source"), expected)

    def test_key_join_reorders_labels_to_signal_order_and_preserves_source_identity(self):
        from sonic.data import raw_bags
        from sonic.fitting import join_labels
        datasets, frames = [], []
        for source in ("human", "synthetic"):
            signal = write_signals(self.directory / f"{source}.json", self.records)
            target = write_targets(self.directory / f"{source}.csv", self.records, reverse=True)
            datasets.append((source, signal, target))
            frames.extend(row for row, _ in raw_bags(signal, source))
        sites = pd.DataFrame(frames).assign(_row=np.arange(len(frames)))
        actual = join_labels(SimpleNamespace(sites=sites), datasets)
        self.assertEqual(actual.transcript_id.tolist(), sites.transcript_id.tolist())
        self.assertEqual(actual.source_id.tolist(), ["human"] * 6 + ["synthetic"] * 6)
        self.assertEqual(actual._row.tolist(), list(range(12)))
        self.assertEqual(actual.label.tolist(), [0, 1, 0, 1, 0, 1] * 2)
        np.testing.assert_array_equal(actual.label_original, [0, .25, 0, .8, 0, 1] * 2)

    def test_missing_extra_and_duplicate_label_keys_fail_without_intersection(self):
        from sonic.data import raw_bags
        from sonic.fitting import join_labels
        signal = write_signals(self.directory / "signals.json", self.records)
        rows = [row for row, _ in raw_bags(signal, "source")]
        sites = pd.DataFrame(rows).assign(_row=np.arange(len(rows)))
        for change in ("missing", "extra", "duplicate"):
            target = write_targets(self.directory / "labels.csv", self.records)
            with target.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, labels = reader.fieldnames, list(reader)
            if change == "missing":
                labels.pop()
            elif change == "extra":
                labels.append({**labels[0], "transcript_id": "absent_from_signals"})
            else:
                labels.append(labels[0])
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(labels)
            with self.subTest(change=change), self.assertRaises(ValueError):
                join_labels(SimpleNamespace(sites=sites), [("source", signal, target)])

    def test_one_class_population_and_conflicting_source_are_rejected(self):
        from sonic.data import raw_bags
        from sonic.fitting import join_labels, read_labels
        signal = write_signals(self.directory / "signals.json", self.records)
        rows = [row for row, _ in raw_bags(signal, "source")]
        sites = pd.DataFrame(rows).assign(_row=np.arange(len(rows)))
        target = write_targets(self.directory / "one-class.csv", self.records, [0.25] * 6)
        with self.assertRaises(ValueError):
            join_labels(SimpleNamespace(sites=sites), [("source", signal, target)])
        labels = pd.read_csv(target).assign(source_id="different_source")
        labels.to_csv(target, index=False)
        with self.assertRaisesRegex(ValueError, "source"):
            read_labels(target, "source")


class TrainingTests(unittest.TestCase):
    def test_training_failure_cleans_staging_and_existing_output_is_protected(self):
        from sonic.fitting import train_model
        with tempfile.TemporaryDirectory(prefix="sonic-failed-training-") as temporary:
            directory = Path(temporary)
            records = signal_records()
            signals = write_signals(directory / "signals.json", records)
            targets = write_targets(directory / "targets.csv", records)
            dataset = [("source", signals, targets)]
            existing = directory / "existing model"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("existing model stays unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_model(dataset, existing)
            self.assertEqual(sentinel.read_text(), "existing model stays unchanged")
            record = records[-1]
            transcript = next(iter(record))
            position = next(iter(record[transcript]))
            sequence = next(iter(record[transcript][position]))
            record[transcript][position][sequence] = record[transcript][position][sequence][:19]
            write_signals(signals, records)
            before = set(directory.iterdir())
            output = directory / "failed fit"
            with self.assertRaisesRegex(ValueError, "20"):
                train_model(dataset, output, temp_dir=directory)
            self.assertFalse(output.exists())
            self.assertEqual(set(directory.iterdir()), before)

    def test_small_two_source_fit_saves_new_portable_model_and_reloads(self):
        from sonic.fitting import train_model
        from sonic.prediction import load_bundle, predict_file
        with tempfile.TemporaryDirectory(prefix="sonic training test ") as temporary:
            directory = Path(temporary)
            datasets = []
            for i, source in enumerate(("human", "synthetic")):
                records = signal_records(seed=742 + i)
                signal = write_signals(directory / f"{source}.json", records)
                targets = write_targets(directory / f"{source}.csv", records,
                                        values=[0, 0.25, 0, 0, 0, 1], reverse=True,
                                        gene_metadata=i == 0)
                datasets.append((source, signal, targets))
            output = directory / "new fit"
            frozen_before = {str(path.relative_to(MODEL)): digest(path)
                             for path in MODEL.rglob("*") if path.is_file()}
            train_model(datasets, output)
            model_dir, bundle = load_bundle(output / "model")
            self.assertEqual(bundle["model"], "embedding_xgboost")
            self.assertEqual(bundle["fixed_steps"], 269)
            self.assertEqual(bundle["weighting"]["counts"], [8, 4])
            self.assertEqual(bundle["weighting"]["mode"], "none")
            self.assertEqual(bundle["weighting"]["weights"], [1.0, 1.0])
            self.assertEqual(bundle["params"]["n_estimators"], 269)
            self.assertEqual(len(bundle["predictors"]), 341)
            _, encoder = load_bundle(model_dir / "encoder")
            self.assertEqual(encoder["fixed_steps"], 18)
            self.assertEqual(encoder["weighting"]["counts"], [8, 4])
            self.assertEqual(encoder["weighting"]["mode"], "none")
            self.assertEqual(encoder["weighting"]["weights"], [1.0, 1.0])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["encoder_epochs"], 18)
            self.assertEqual(manifest["tree_count"], 269)
            reference = json.loads((model_dir / "motif_reference.json").read_text())
            self.assertEqual(reference["fitting_sites"], 12)
            self.assertNotEqual(digest(model_dir / "encoder/weights.pt"),
                                digest(MODEL / "encoder/weights.pt"))
            # Reload a copy after removing original run location and input targets.
            relocated = directory / "relocated"
            (output / "model").rename(relocated)
            for _, _, targets in datasets:
                targets.unlink()
            prediction = directory / "new-model-predictions.csv"
            self.assertEqual(predict_file(datasets[0][1], prediction, model_path=relocated), 6)
            keys, _ = read_predictions(prediction)
            self.assertEqual(keys, site_keys(signal_records()))
            self.assertEqual(frozen_before,
                             {str(path.relative_to(MODEL)): digest(path)
                              for path in MODEL.rglob("*") if path.is_file()})

    def test_train_cli_help_and_missing_inputs_are_actionable(self):
        with tempfile.TemporaryDirectory(prefix="sonic-train-cli-") as temporary:
            directory = Path(temporary)
            help_result = subprocess.run([sys.executable, str(ROOT / "train.py"), "--help"],
                                         cwd=directory, capture_output=True, text=True,
                                         timeout=30, check=False)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--dataset", help_result.stdout)
            self.assertIn("--output", help_result.stdout)
            output = directory / "should-not-exist"
            result = subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--dataset", "source",
                 str(directory / "missing.json"), str(directory / "missing.csv"),
                 "--output", str(output)],
                cwd=directory, capture_output=True, text=True, timeout=30, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing", result.stderr.lower())
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_truncated_compressed_labels_fail_without_traceback_or_output(self):
        with tempfile.TemporaryDirectory(prefix="sonic-truncated-labels-") as temporary:
            directory = Path(temporary)
            records = signal_records()
            signal = write_signals(directory / "signals.json", records)
            labels = write_targets(directory / "data.info.labelled", records)
            compressed = directory / "data.info.labelled.gz"
            compressed.write_bytes(gzip.compress(labels.read_bytes())[:-8])
            output = directory / "failed-model"
            result = subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--dataset", "source",
                 str(signal), str(compressed), "--output", str(output)],
                cwd=directory, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Training failed:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_corrupt_compressed_inputs_fail_without_traceback_or_partial_output(self):
        with tempfile.TemporaryDirectory(prefix="sonic-corrupt-training-input-") as temporary:
            directory = Path(temporary)
            records = signal_records()
            signal = write_signals(directory / "signals.json", records)
            labels = write_targets(directory / "labels.csv", records)
            for kind, original in (("signals", signal), ("labels", labels)):
                with self.subTest(kind=kind):
                    compressed = directory / (original.name + ".gz")
                    corrupt = bytearray(gzip.compress(original.read_bytes()))
                    # The fixed ten-byte gzip header precedes the DEFLATE data.
                    corrupt[10] = 0xff
                    compressed.write_bytes(corrupt)
                    output = directory / ("failed-model-" + kind)
                    source = compressed if kind == "signals" else signal
                    targets = compressed if kind == "labels" else labels
                    before = set(directory.iterdir())
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "train.py"), "--dataset", "source",
                         str(source), str(targets), "--output", str(output),
                         "--temp-dir", str(directory)],
                        cwd=directory, capture_output=True, text=True, timeout=90, check=False)
                    self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
                    self.assertIn("Training failed:", result.stderr)
                    self.assertIn("decompress", result.stderr.lower())
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output.exists())
                    self.assertEqual(set(directory.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
