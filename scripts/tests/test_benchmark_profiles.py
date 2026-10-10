from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SUMMARY_SCRIPT = Path(__file__).parents[1] / "benchmark_summary.py"


class BenchmarkProfileTests(unittest.TestCase):
    def test_summary_selects_requested_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "timings.csv"
            csv_path.write_text(
                "benchmark_name,scale,prover_time_s\n"
                "fibonacci,20,1\n"
                "fibonacci_akita,20,2\n"
                "fibonacci_akita_w2r2,20,3\n"
                "fibonacci_akita_w4r2,20,4\n"
                "fibonacci_akita_w8r2,20,5\n"
            )
            for options, expected in [
                ([], "1.00s"),
                (["--protocol", "akita"], "2.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w2r2"], "3.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w4r2"], "4.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w8r2"], "5.00s"),
            ]:
                with self.subTest(options=options):
                    result = subprocess.run(
                        [sys.executable, str(SUMMARY_SCRIPT), "--csv", str(csv_path),
                         "--metric", "prover_time_s", *options],
                        check=True, capture_output=True, text=True,
                    )
                    rows = [line.split("|") for line in result.stdout.splitlines()
                            if line.strip().startswith("2^20")]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][1].strip(), expected)

    def test_chunked_profile_requires_akita_protocol(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SUMMARY_SCRIPT), "--akita-chunk-profile", "w4r2"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --protocol akita", result.stderr)


if __name__ == "__main__":
    unittest.main()
