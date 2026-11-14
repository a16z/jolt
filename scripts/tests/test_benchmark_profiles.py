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
                "fibonacci_akita_w16r4,20,6\n"
            )
            for options, expected in [
                ([], "1.00s"),
                (["--protocol", "akita"], "2.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w2r2"], "3.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w4r2"], "4.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w8r2"], "5.00s"),
                (["--protocol", "akita", "--akita-chunk-profile", "w16r4"], "6.00s"),
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

    def test_summary_keeps_kernel_backends_separate(self) -> None:
        reference_rows = [
            "fibonacci,20,100,reference\n",
            "sha2-chain,20,200,\n",
        ]
        optimized_rows = [
            "fibonacci,20,1,optimized\n",
            "sha2-chain,20,2,optimized\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "timings.csv"
            for rows in [
                reference_rows + optimized_rows,
                optimized_rows + reference_rows,
                [reference_rows[0], optimized_rows[0],
                 optimized_rows[1], reference_rows[1]],
            ]:
                csv_path.write_text(
                    "benchmark_name,scale,prover_time_s,backend\n" + "".join(rows)
                )
                for options, backend, expected in [
                    ([], "reference", ["100.00s", "200.00s"]),
                    (["--backend", "optimized"], "optimized", ["1.00s", "2.00s"]),
                ]:
                    with self.subTest(rows=rows, backend=backend):
                        result = subprocess.run(
                            [sys.executable, str(SUMMARY_SCRIPT), "--csv", str(csv_path),
                             "--metric", "prover_time_s", *options],
                            check=True, capture_output=True, text=True,
                        )
                        self.assertIn(backend, result.stdout.splitlines()[1])
                        table_rows = [
                            line.split("|") for line in result.stdout.splitlines()
                            if line.strip().startswith("2^20")
                        ]
                        self.assertEqual(len(table_rows), 1)
                        self.assertEqual(
                            [cell.strip() for cell in table_rows[0][1:3]], expected
                        )

    def test_summary_uses_latest_matching_backend_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "timings.csv"
            csv_path.write_text(
                "benchmark_name,scale,prover_time_s,backend\n"
                "fibonacci_akita_w4r2,20,3,optimized\n"
                "fibonacci_akita_w4r2,20,4,optimized\n"
                "fibonacci_akita_w4r2,20,,optimized\n"
                "fibonacci_akita_w4r2,20,100,reference\n"
                "fibonacci_akita_w2r2,20,200,optimized\n"
            )
            result = subprocess.run(
                [sys.executable, str(SUMMARY_SCRIPT), "--csv", str(csv_path),
                 "--metric", "prover_time_s", "--protocol", "akita",
                 "--akita-chunk-profile", "w4r2", "--backend", "optimized"],
                check=True, capture_output=True, text=True,
            )
            rows = [line.split("|") for line in result.stdout.splitlines()
                    if line.strip().startswith("2^20")]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1].strip(), "4.00s")

    def test_chunked_profile_requires_akita_protocol(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SUMMARY_SCRIPT), "--akita-chunk-profile", "w4r2"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --protocol akita", result.stderr)


if __name__ == "__main__":
    unittest.main()
