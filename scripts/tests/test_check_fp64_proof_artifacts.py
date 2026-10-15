import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "check_fp64_proof_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_fp64_proof_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckFp64ProofArtifactsTests(unittest.TestCase):
    def test_aarch64_targets_have_separate_exact_sequences(self) -> None:
        self.assertNotEqual(
            CHECKER.AARCH64["darwin"]["add"],
            CHECKER.AARCH64["linux"]["add"],
        )
        for target_os in ("darwin", "linux"):
            for operation in ("add", "sub", "mul"):
                self.assertEqual(CHECKER.AARCH64[target_os][operation][-1], 0xD65F03C0)

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
        function, trailing = CHECKER.through_ret(disassembly, "jolt_fp64_sub_asm")
        self.assertEqual(function, bytes.fromhex("48 29 f7 c3"))
        self.assertEqual(trailing, [(b"\x90", "nop")])

    def test_constructs_only_the_expected_darwin_frame(self) -> None:
        body = bytes.fromhex("48 29 f7 c3")
        self.assertEqual(
            CHECKER.darwin_frame(body),
            bytes.fromhex("55 48 89 e5 48 29 f7 5d c3"),
        )

    def test_post_return_policy_rejects_non_nop_instruction(self) -> None:
        with self.assertRaises(SystemExit):
            CHECKER.require_post_return(
                "test",
                [(b"\x90", "nop"), (b"\x31\xc0", "xorl %eax, %eax")],
                "nop-padding-only",
            )

    def test_post_return_policy_rejects_any_linux_trailing_instruction(self) -> None:
        with self.assertRaises(SystemExit):
            CHECKER.require_post_return("test", [(b"\x90", "nop")], "none")

    def test_post_return_policy_accepts_only_exact_int3_padding(self) -> None:
        CHECKER.require_post_return(
            "test",
            [(b"\xcc", "int3"), (b"\xcc", "int3")],
            "int3-padding-only",
        )
        for trailing in (
            [(b"\x90", "nop")],
            [(b"\xcc", "nop")],
            [(b"\xcd\x03", "int $3")],
        ):
            with self.assertRaises(SystemExit):
                CHECKER.require_post_return(
                    "test",
                    trailing,
                    "int3-padding-only",
                )

    def test_format_marker_must_match(self) -> None:
        with self.assertRaises(SystemExit):
            CHECKER.require_format(
                "test",
                "binary: file format mach-o arm64",
                "file format elf64-littleaarch64",
            )

    def test_require_rejects_drift(self) -> None:
        with self.assertRaises(SystemExit):
            CHECKER.require("test", b"old", b"new")


if __name__ == "__main__":
    unittest.main()
