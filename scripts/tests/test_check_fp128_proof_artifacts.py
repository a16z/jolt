import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "check_fp128_proof_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_fp128_proof_artifacts", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckFp128ProofArtifactsTests(unittest.TestCase):
    def test_parses_linux_and_macos_symbol_names(self) -> None:
        disassembly = """
0000000000000000 <_jolt_fp128_add_asm>:
       0: eb020005      subs x5, x0, x2
       4: d65f03c0      ret
0000000000000008 <jolt_fp128_sub_asm>:
       8: 128b0104      mov w4, #-0x5809
"""
        self.assertEqual(
            CHECKER.parse_symbol_words(disassembly),
            {
                "jolt_fp128_add_asm": [0xEB020005, 0xD65F03C0],
                "jolt_fp128_sub_asm": [0x128B0104],
            },
        )

    def test_word_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "instruction mismatch"):
            CHECKER.require_words("test", [1], [2])

    def test_parses_x86_bytes_and_macos_symbol_name(self) -> None:
        disassembly = """
0000000000000000 <_jolt_fp128_sub_asm>:
       0: 48 29 d7                      subq %rdx, %rdi
       3: 48 83 de 00                   sbbq $0x0, %rsi
       7: c3                            retq
"""
        self.assertEqual(
            CHECKER.parse_symbol_bytes(disassembly),
            {"jolt_fp128_sub_asm": bytes.fromhex("48 29 d7 48 83 de 00 c3")},
        )
        self.assertEqual(
            CHECKER.parse_symbol_byte_instructions(disassembly),
            {
                "jolt_fp128_sub_asm": [
                    bytes.fromhex("48 29 d7"),
                    bytes.fromhex("48 83 de 00"),
                    bytes.fromhex("c3"),
                ]
            },
        )

    def test_byte_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "byte mismatch"):
            CHECKER.require_bytes("test", bytes([1]), bytes([2]))

    def test_symbol_is_trimmed_at_decoded_ret(self) -> None:
        instructions = [bytes.fromhex("55"), bytes.fromhex("c3"), bytes.fromhex("90")]
        self.assertEqual(
            CHECKER.instructions_through_ret("test", instructions),
            bytes.fromhex("55 c3"),
        )
        with self.assertRaisesRegex(SystemExit, "no decoded ret"):
            CHECKER.instructions_through_ret("test", [bytes.fromhex("90")])

if __name__ == "__main__":
    unittest.main()
