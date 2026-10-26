"""Shell entry points: portable paths, retained examples and failure cleanup.

These fixtures replace model computation with tiny Python commands. Installation
and numerical correctness are checked separately; these tests never install pip
packages or depend on a particular user's existing virtual environment.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sonic launcher tests ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.release = self.directory / "checkout with spaces"
        self.release.mkdir()
        for name in ("run.sh", "install.sh"):
            shutil.copy2(ROOT / name, self.release / name)
        self.caller = self.directory / "caller with spaces"
        self.caller.mkdir()
        self.scratch = self.directory / "temporary outputs"
        self.scratch.mkdir()
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment["TMPDIR"] = str(self.scratch)

    def invoke(self, *args, script="run.sh"):
        return subprocess.run(
            ["bash", str(self.release / script), *args], cwd=self.caller,
            env=self.environment, text=True, capture_output=True, timeout=15,
            check=False)

    def interpreter(self):
        """Provide an executable for dispatch, without pretending to install."""
        executable = self.release / ".venv/bin/python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)

    def example(self, *, fail_prediction=False, incorrect_score=False):
        self.interpreter()
        examples = self.release / "examples"
        examples.mkdir()
        shutil.copy2(ROOT / "examples/verify.py", examples / "verify.py")
        (examples / "test.json").write_text("fixture input", encoding="utf-8")
        expected = "transcript_id,transcript_position,score\nexample,7,0.25\n"
        (examples / "expected_predictions.csv").write_text(expected, encoding="utf-8")
        actual = expected.replace("0.25", "0.75") if incorrect_score else expected
        (self.release / "predict.py").write_text(
            "import argparse\nfrom pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--input', type=Path, required=True)\n"
            "parser.add_argument('--output', type=Path, required=True)\n"
            "args = parser.parse_args()\n"
            "assert args.input.read_text() == 'fixture input'\n"
            f"args.output.write_text({actual!r})\n"
            f"raise SystemExit({7 if fail_prediction else 0})\n", encoding="utf-8")

    def test_help_and_invalid_commands_work_without_installation(self):
        for arguments in (("--help",), ("-h",), ("help",)):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("predict", result.stdout)
                self.assertIn("train", result.stdout)
        for arguments in ((), ("unknown-command",), ("test", "unexpected")):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertFalse((self.release / ".venv").exists())

    def test_missing_environment_points_to_installer(self):
        for command in ("predict", "train", "test"):
            with self.subTest(command=command):
                result = self.invoke(command)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("install.sh", result.stderr + result.stdout)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_prediction_and_training_preserve_arguments_cwd_and_exit_status(self):
        self.interpreter()
        receipt = self.caller / "command receipt.json"
        self.environment["LAUNCHER_RECEIPT"] = str(receipt)
        command = (
            "import json, os, sys\nfrom pathlib import Path\n"
            "Path(os.environ['LAUNCHER_RECEIPT']).write_text(json.dumps({"
            "'arguments': sys.argv[1:], 'cwd': os.getcwd(), 'script': __file__}))\n"
            "raise SystemExit(7)\n")
        arguments = ["--input", "relative input with spaces.json.gz", "--output",
                     "relative output.csv", "--literal", "$(not-a-command)", ""]
        for name in ("predict", "train"):
            with self.subTest(command=name):
                (self.release / f"{name}.py").write_text(command, encoding="utf-8")
                result = self.invoke(name, *arguments)
                self.assertEqual(result.returncode, 7, result.stderr)
                actual = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(actual["arguments"], arguments)
                self.assertEqual(Path(actual["cwd"]), self.caller)
                self.assertEqual(Path(actual["script"]), self.release / f"{name}.py")

    def test_bundled_example_repeats_with_distinct_inspectable_outputs(self):
        self.example()
        retained = []
        for _ in range(2):
            result = self.invoke("test")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS", result.stdout)
            output_lines = [line for line in result.stdout.splitlines()
                            if line.startswith("Verified example CSV: ")]
            self.assertEqual(len(output_lines), 1, result.stdout)
            output = Path(output_lines[0].removeprefix("Verified example CSV: "))
            self.assertTrue(output.is_absolute())
            self.assertEqual(output.parent.parent, self.release / "outputs")
            self.assertNotIn(output, retained)
            retained.append(output)
            expected = (self.release / "examples/expected_predictions.csv").read_bytes()
            for previous in retained:
                self.assertEqual(previous.read_bytes(), expected)
            self.assertEqual(len(list((self.release / "outputs").iterdir())), len(retained))
            self.assertEqual(list(self.scratch.iterdir()), [])
            self.assertEqual(list(self.caller.iterdir()), [])

    def test_prediction_failure_propagates_and_cleans_partial_example_output(self):
        self.example(fail_prediction=True)
        result = self.invoke("test")
        self.assertEqual(result.returncode, 7, result.stderr + result.stdout)
        self.assertNotIn("PASS", result.stdout)
        self.assertEqual(list((self.release / "outputs").iterdir()), [])
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_verification_failure_propagates_and_cleans_example_output(self):
        self.example(incorrect_score=True)
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Verification failed", result.stderr)
        self.assertEqual(list((self.release / "outputs").iterdir()), [])
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_example_reports_unavailable_output_directory_without_overwriting(self):
        self.example()
        blocked = self.release / "outputs"
        blocked.write_text("keep this file", encoding="utf-8")
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot create output directory", result.stderr)
        self.assertNotIn("PASS", result.stdout)
        self.assertEqual(blocked.read_text(encoding="utf-8"), "keep this file")

    def test_installer_help_and_unknown_arguments_do_not_create_environment(self):
        result = self.invoke("--help", script="install.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.invoke("--unknown", script="install.sh")
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertFalse((self.release / ".venv").exists())

    def test_installer_preserves_existing_non_environment_directory(self):
        existing = self.release / ".venv"
        existing.mkdir()
        sentinel = existing / "user-file.txt"
        sentinel.write_text("keep this file", encoding="utf-8")
        result = self.invoke(script="install.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this file")
        self.assertEqual(list(existing.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
