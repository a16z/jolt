import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import check_field_proof_final_binary as checker


class FinalBinaryProofSequenceTests(unittest.TestCase):
    def test_aarch64_matches_only_at_instruction_boundaries(self) -> None:
        patterns = checker.expected_patterns("aarch64")
        expected = patterns["fp64"]["sub_linux"]
        words = [
            int.from_bytes(expected[index : index + 4], "little")
            for index in range(0, len(expected), 4)
        ]
        disassembly = "0000000000001000 <_verified>:\n" + "\n".join(
            f"    {0x1000 + 4 * index:x}: {word:08x} instruction"
            for index, word in enumerate(words)
        )
        symbols = checker.parse_disassembly(disassembly, "aarch64")
        matches = checker.find_matches(
            symbols, {"fp64": {"sub_linux": expected}}
        )
        self.assertEqual(
            matches,
            [checker.Match("fp64", "sub_linux", "verified", "0x1000")],
        )

    def test_aarch64_patterns_include_both_proved_targets(self) -> None:
        operations = checker.expected_patterns("aarch64")["fp64"]
        self.assertEqual(
            set(operations),
            {
                "add_darwin",
                "sub_darwin",
                "mul_darwin",
                "add_linux",
                "sub_linux",
                "mul_linux",
            },
        )

    def test_x86_rejects_sequence_starting_inside_instruction(self) -> None:
        expected = bytes.fromhex("11 22")
        disassembly = """
0000000000002000 <candidate>:
    2000: aa 11 instruction
    2002: 22 bb instruction
"""
        symbols = checker.parse_disassembly(disassembly, "x86_64")
        matches = checker.find_matches(symbols, {"test": {"op": expected}})
        self.assertEqual(matches, [])

    def test_x86_accepts_exact_consecutive_instructions(self) -> None:
        expected = bytes.fromhex("48 29 f7 48 8d 47 c5")
        disassembly = """
0000000000003000 <candidate>:
    3000: 48 29 f7 subq %rsi, %rdi
    3003: 48 8d 47 c5 leaq -59(%rdi), %rax
    3007: c3 retq
"""
        symbols = checker.parse_disassembly(disassembly, "x86_64")
        matches = checker.find_matches(symbols, {"test": {"op": expected}})
        self.assertEqual(
            matches,
            [checker.Match("test", "op", "candidate", "0x3000")],
        )

    def test_x86_joins_gnu_objdump_byte_continuations(self) -> None:
        disassembly = """
0000000000003000 <candidate>:
    3000: c4 62 b3 f6 d7 c4 62
    3007: fb f6 de             mulx %rsi,%rax,%rbx
    300a: c3                   ret
"""
        instructions = checker.parse_disassembly(disassembly, "x86_64")["candidate"]
        self.assertEqual(
            instructions[0].encoding,
            bytes.fromhex("c4 62 b3 f6 d7 c4 62 fb f6 de"),
        )
        self.assertEqual(instructions[0].text, "mulx %rsi,%rax,%rbx")

    def test_required_family_needs_add_sub_and_multiplication(self) -> None:
        incomplete = [checker.Match("fp128", "add", "f", "0x1")]
        with self.assertRaises(SystemExit):
            checker.validate_required("fp128", incomplete)
        complete = [
            checker.Match("fp128", "add", "f", "0x1"),
            checker.Match("fp128", "sub", "f", "0x2"),
            checker.Match("fp128", "mul_bmi2_adx", "f", "0x3"),
        ]
        checker.validate_required("fp128", complete)
        target_named = [
            checker.Match("fp64", "add_linux", "f", "0x1"),
            checker.Match("fp64", "sub_linux", "f", "0x2"),
            checker.Match("fp64", "mul_linux", "f", "0x3"),
        ]
        checker.validate_required("fp64", target_named)

    def test_aarch64_template_accepts_consistent_register_renaming(self) -> None:
        expected = checker.parse_disassembly(
            """
0000000000000000 <proof>:
    0: ab020005 adds x5, x0, x2
    4: ba030026 adcs x6, x1, x3
    8: 9a851100 csel x0, x8, x5, ne
""",
            "aarch64",
        )["proof"]
        actual = checker.parse_disassembly(
            """
0000000000001000 <linked>:
    1000: ab0a010c adds x12, x8, x10
    1004: ba0b012d adcs x13, x9, x11
    1008: 9a8c01e8 csel x8, x15, x12, ne
""",
            "aarch64",
        )["linked"]
        self.assertEqual(
            checker.unify_instruction_templates(expected, actual, "aarch64"),
            {
                "x0": "x8",
                "x1": "x9",
                "x2": "x10",
                "x3": "x11",
                "x5": "x12",
                "x6": "x13",
                "x8": "x15",
            },
        )

    def test_template_rejects_noninjective_register_mapping(self) -> None:
        expected = checker.parse_disassembly(
            """
0000000000000000 <proof>:
    0: 48 01 d7 addq %rdx, %rdi
    3: 48 11 ce adcq %rcx, %rsi
""",
            "x86_64",
        )["proof"]
        actual = checker.parse_disassembly(
            """
0000000000001000 <linked>:
    1000: 48 01 f8 addq %rdi, %rax
    1003: 48 11 f0 adcq %rsi, %rax
""",
            "x86_64",
        )["linked"]
        self.assertIsNone(
            checker.unify_instruction_templates(expected, actual, "x86_64")
        )

    def test_x86_numbered_register_aliases_have_one_family(self) -> None:
        self.assertEqual(checker.x86_register("r9"), ("r9", "64"))
        self.assertEqual(checker.x86_register("r9d"), ("r9", "32"))
        self.assertEqual(checker.x86_register("r15b"), ("r15", "8"))


if __name__ == "__main__":
    unittest.main()
