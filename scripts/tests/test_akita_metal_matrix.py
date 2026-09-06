"""Acceptance and aggregation contracts for the portable Metal matrix runner."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("matrix", Path(__file__).resolve().parents[1] / "akita_metal_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def raw_result(workload="fibonacci", seconds="2", trace=10000000, scale=24):
    return ("PROOF_VERIFIED backend=metal value=true\n"
            f"MATRIX_TIMING name={workload} scale={scale} backend=metal prove_s={seconds} "
            f"trace_len={trace} padded_len={1 << scale}\n"
            "1073741824 maximum resident set size\n0 swaps\n")


class MatrixContract(unittest.TestCase):
    def test_padded_and_actual_rates_are_distinct(self):
        row = matrix.parse_result(raw_result(), "fibonacci", 24)
        self.assertEqual(row["padded_mhz"], 8.388608)
        self.assertEqual(row["actual_mhz"], 5)
        self.assertEqual(row["rss_gib"], 1)

    def test_invalid_observations_fail_closed(self):
        raw = raw_result()
        cases = [raw.replace("value=true", "value=false"), raw + "PROOF_VERIFIED backend=metal value=true\n",
                 raw + "watchdog abort\n", raw.replace("0 swaps", "1 swaps"),
                 raw.replace("name=fibonacci", "name=btreemap"), raw.replace("scale=24", "scale=25"),
                 raw.replace("padded_len=16777216", "padded_len=33554432"),
                 raw.replace("trace_len=10000000", "trace_len=8388608"),
                 raw.replace("trace_len=10000000", "trace_len=16777217"),
                 raw.replace("1073741824 maximum", f"{matrix.RSS_LIMIT} maximum"),
                 raw + "MATRIX_TIMING name=fibonacci\n"]
        cases += [raw_result(seconds=value) for value in ("0", "-1", "nan", "inf")]
        for case in cases:
            with self.subTest(raw=case), self.assertRaises(ValueError):
                matrix.parse_result(case, "fibonacci", 24)

    def test_blake_digest_and_iteration_identity(self):
        # Frozen independent hashlib result from the verified 80k observation.
        digest = ("d6c7a7ce7dcefc936a44d379c49bd908a0f6ae59c85eb134443d708be91aa7f09a"
                  "71e2dfac6facb18646405b8142b9c6163272bf039550a2f0fb2dad5ba4e14f")
        raw = (f"BLAKE2B_CHAIN_CHECK iterations=80000 digest={digest} value=true\n"
               + raw_result("blake2b-chain", trace=166883688, scale=28))
        matrix.parse_result(raw, "blake2b-chain", 28)
        for wrong in (raw.replace("iterations=80000", "iterations=79999"), raw.replace(digest, "00" * 64)):
            with self.assertRaises(ValueError):
                matrix.parse_result(wrong, "blake2b-chain", 28)

    def test_mean_of_rates_and_no_partial_or_duplicate_pass(self):
        rows = [dict(workload=name, scale=28, padded_mhz=rate)
                for name, rate in zip(matrix.WORKLOADS, (4, 8, 12, 16))]
        self.assertEqual(matrix.mean_rates(rows)[0]["measured_mean_mhz"], 10)
        self.assertTrue(matrix.mean_rates(rows)[0]["measured_10mhz_pass"])
        self.assertEqual(matrix.mean_rates(rows[:-1]), [])
        self.assertEqual(matrix.mean_rates(rows + [rows[0]]), [])
        rows[0]["padded_mhz"] = 0
        self.assertFalse(matrix.mean_rates(rows)[0]["measured_10mhz_pass"])

    def test_resume_rejects_changed_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("cells").mkdir()
            raw = root / "cells/t24_fibonacci.out"
            raw.write_text(raw_result())
            row = matrix.parse_result(raw.read_text(), "fibonacci", 24)
            row.update(raw="cells/t24_fibonacci.out", raw_sha256=matrix.digest(raw))
            root.joinpath("cells/t24_fibonacci.json").write_text(json.dumps(row))
            self.assertEqual(len(matrix.Study(root).load_results()), 1)
            raw.write_text(raw_result(seconds="1"))
            with self.assertRaises(RuntimeError):
                matrix.Study(root).load_results()


if __name__ == "__main__":
    unittest.main()
