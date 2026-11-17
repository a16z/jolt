import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).parents[1]
SUMMARY_SCRIPT = SCRIPTS / "benchmark_summary.py"
PLOT_SCRIPT = SCRIPTS / "plot_benchmarks.py"
MEMORY_PLOT_SCRIPT = SCRIPTS / "plot_memory_usage.py"
PLOTLY_STUB = Path(__file__).parent / "fixtures" / "plotly_stub"
CSV_HEADER = ("benchmark_name,scale,prover_time_s,trace_length,proving_hz,"
              "proof_size,proof_size_compressed,backend\n")


def run_report_script(script: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *options],
        env={**os.environ, "PYTHONPATH": str(PLOTLY_STUB)},
        capture_output=True, text=True,
    )


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

    def test_reports_select_any_backend_named_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "timings.csv"
            csv_path.write_text(
                CSV_HEADER
                + "fibonacci,20,100,1048576,10485.76,1000,900,reference\n"
                + "fibonacci,21,1,2097152,2097152,1000,900,metal\n"
            )
            summary = run_report_script(
                SUMMARY_SCRIPT, "--csv", str(csv_path), "--backend", "metal")
            self.assertEqual(summary.returncode, 0, summary.stderr)
            self.assertEqual(
                [line.split("|")[0].strip() for line in summary.stdout.splitlines()
                 if line.strip().startswith("2^")],
                ["2^21"],
            )

            plot = run_report_script(
                PLOT_SCRIPT, "--csv", str(csv_path), "--backend", "metal",
                "--plot-type", "speed", "--output-dir", directory)
            self.assertEqual(plot.returncode, 0, plot.stderr)
            traces = json.loads((Path(directory) / "benchmark_plot.html").read_text())
            self.assertEqual([trace["x"] for trace in traces], [[21]])

    def test_reports_reject_backend_absent_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "timings.csv"
            csv_path.write_text(
                CSV_HEADER
                + "fibonacci,20,1,1048576,1048576,1000,900,optimized\n"
                + "fibonacci,20,2,1048576,524288,1000,900,metal\n"
            )
            for script, options in [
                (SUMMARY_SCRIPT, []),
                (PLOT_SCRIPT, ["--output-dir", directory]),
            ]:
                with self.subTest(script=script.name):
                    result = run_report_script(
                        script, "--csv", str(csv_path), *options)
                    self.assertEqual(result.returncode, 1)
                    for expected in ["'reference'", str(csv_path), "metal, optimized"]:
                        self.assertIn(expected, result.stderr)

    def test_memory_plot_keeps_kernel_backends_in_separate_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for run, backend, peak in [
                ("latest_modular_fibonacci_20", "reference", 18.5),
                ("latest_modular_fibonacci_20_optimized", "optimized", 2.25),
            ]:
                run_dir = Path(directory) / run
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(json.dumps({
                    "run": {"workload": "fibonacci", "scale_log2": 20,
                            "backend": backend},
                    "peak_rss_gib": peak,
                }))
            result = run_report_script(
                MEMORY_PLOT_SCRIPT, "--traces-dir", directory,
                "--output-dir", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            traces = json.loads(
                (Path(directory) / "memory_usage_plot.html").read_text())
            self.assertEqual(
                {trace["name"]: (trace["x"], trace["y"]) for trace in traces},
                {"Fibonacci (reference)": ([20], [18.5]),
                 "Fibonacci (optimized)": ([20], [2.25])},
            )

    def test_chunked_profile_requires_akita_protocol(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SUMMARY_SCRIPT), "--akita-chunk-profile", "w4r2"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --protocol akita", result.stderr)


if __name__ == "__main__":
    unittest.main()
