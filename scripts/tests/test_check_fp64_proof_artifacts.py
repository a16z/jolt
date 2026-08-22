import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "check_fp64_proof_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_fp64_proof_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckFp64ProofArtifactsTests(unittest.TestCase):
    def test_parses_aarch64_words_and_macos_symbol(self) -> None:
        disassembly = """
0000000000000000 <_jolt_fp64_sub_asm>:
       0: eb010008     subs x8, x0, x1
       4: d65f03c0     ret
"""
        self.assertEqual(
            CHECKER.parse_words(disassembly, "jolt_fp64_sub_asm"),
            [0xEB010008, 0xD65F03C0],
        )

    def test_parses_x86_bytes(self) -> None:
        disassembly = """
0000000000000000 <jolt_fp64_sub_asm>:
       0: 48 29 f7                sub rdi, rsi
       3: c3                      ret
"""
        self.assertEqual(
            CHECKER.parse_bytes(disassembly, "jolt_fp64_sub_asm"),
            bytes.fromhex("48 29 f7 c3"),
        )

    def test_x86_symbol_stops_at_return(self) -> None:
        disassembly = """
0000000000000000 <jolt_fp64_sub_asm>:
       0: 48 29 f7                sub rdi, rsi
       3: c3                      ret
       4: 90                      nop
"""
        self.assertEqual(
            CHECKER.through_ret(disassembly, "jolt_fp64_sub_asm"),
            bytes.fromhex("48 29 f7 c3"),
        )

    def test_normalizes_only_the_expected_darwin_frame(self) -> None:
        body = bytes.fromhex("48 29 f7 c3")
        framed = bytes.fromhex("55 48 89 e5 48 29 f7 5d c3")
        self.assertEqual(CHECKER.normalize_darwin_frame(framed), body)
        self.assertEqual(CHECKER.normalize_darwin_frame(body), body)

    def test_require_rejects_drift(self) -> None:
        with self.assertRaises(SystemExit):
            CHECKER.require("test", b"old", b"new")


if __name__ == "__main__":
    unittest.main()
