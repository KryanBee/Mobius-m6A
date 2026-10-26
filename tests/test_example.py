"""Check the public example verifier's CSV and numerical promises."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HEADER = "transcript_id,transcript_position,score\n"


class ExampleVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sonic-example-verifier-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.verifier = self.directory / "verify.py"
        shutil.copy2(ROOT / "examples/verify.py", self.verifier)

    def verify(self, actual, expected=HEADER + "example,7,0.25\n"):
        (self.directory / "expected_predictions.csv").write_text(expected, encoding="utf-8")
        output = self.directory / "predictions.csv"
        output.write_text(actual, encoding="utf-8")
        return subprocess.run([sys.executable, str(self.verifier), str(output)],
                              text=True, capture_output=True, timeout=15, check=False)

    def test_success_reports_shape_range_and_expected_values(self):
        content = HEADER + "example,7,0.25\n"
        result = self.verify(content)
        self.assertEqual(result.returncode, 0, result.stderr)
        for detail in ("PASS", "three-column CSV", "1 sites", "finite scores in [0, 1]",
                       "identifiers/order", "absolute tolerance 1e-8"):
            self.assertIn(detail, result.stdout)

    def test_scores_outside_range_fail_even_within_reference_tolerance(self):
        for actual, reference in ((-1e-9, 0), (1 + 1e-9, 1), ("nan", .25), ("inf", .25)):
            with self.subTest(score=actual):
                result = self.verify(HEADER + f"example,7,{actual}\n",
                                     HEADER + f"example,7,{reference}\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite and in [0, 1]", result.stderr)
                self.assertNotIn("PASS", result.stdout)

    def test_csv_shape_site_identity_and_score_mismatches_fail(self):
        bad_outputs = (
            "transcript_id,transcript_position,score,extra\nexample,7,0.25,extra\n",
            HEADER + "example,7,0.25,extra\n",
            HEADER + "example,7\n",
            HEADER,
            HEADER + "another,7,0.25\n",
            HEADER + "example,7,0.75\n",
        )
        for content in bad_outputs:
            with self.subTest(content=content):
                result = self.verify(content)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Verification failed", result.stderr)
                self.assertNotIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
