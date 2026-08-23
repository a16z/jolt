#!/usr/bin/env python3
"""Ratchet proved Fp64 object bytes against compiled production witnesses."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp64_certified_matrix import (  # noqa: E402
    DEFAULT_MATRIX,
    load_matrix,
    profile_by_id,
    target_by_id,
    validate_matrix,
)


AARCH64 = {
    "darwin": {
        "add": [
            0xAB000028, 0x52800769, 0x9A9F2129, 0x8B080128,
            0x9100ED09, 0xEB08013F, 0x9A883120, 0xD65F03C0,
        ],
        "sub": [0xEB010008, 0x92800749, 0x9A9F3129, 0x8B090100, 0xD65F03C0],
        "mul": [
            0x9B007C28, 0x9BC07C29, 0x5280076A, 0x9BCA7D2B,
            0xCB09016B, 0x9B0A7D2C, 0xAB080188, 0x9A090169,
            0x9BCA7D2B, 0xCB09016B, 0x9B0A7D2A, 0xAB080148,
            0x9A090169, 0x9100ED0A, 0xB100ED1F, 0xFA1F013F,
            0x9A8A3100, 0xD65F03C0,
        ],
    },
    "linux": {
        "add": [
            0x52800768, 0xAB000029, 0x9A9F2108, 0x8B090108,
            0x9100ED09, 0xEB08013F, 0x9A883120, 0xD65F03C0,
        ],
        "sub": [0x92800748, 0xEB010009, 0x9A9F3108, 0x8B080120, 0xD65F03C0],
        "mul": [
            0x9BC07C29, 0x52800768, 0x9B007C2A, 0x9BC87D2B,
            0x9B087D2C, 0xCB09016B, 0xAB0A018A, 0x9A090169,
            0x9BC87D2B, 0x9B087D28, 0xCB09016B, 0xAB0A0108,
            0x9A090169, 0xB100ED1F, 0x9100ED0A, 0xFA1F013F,
            0x9A8A3100, 0xD65F03C0,
        ],
    },
}

X86_64 = {
    "add": bytes.fromhex(
        "48 01 f7 48 8d 47 3b 48 0f 43 c7 48 8d 48 3b "
        "48 39 c1 48 0f 42 c1 c3"
    ),
    "sub": bytes.fromhex("48 29 f7 48 8d 47 c5 48 0f 43 c7 c3"),
    "mul": bytes.fromhex(
        "48 89 f0 48 f7 e7 48 89 c1 48 89 d6 41 b9 3b 00 00 00 "
        "48 89 d0 49 f7 e1 48 89 c7 49 89 d0 49 29 f0 48 01 cf "
        "49 11 f0 4c 89 c0 49 f7 e1 4c 29 c2 48 01 f8 4c 11 c2 "
        "48 89 c1 48 83 e9 c5 48 83 da 00 48 0f 43 c1 c3"
    ),
    "mul_bmi2": bytes.fromhex(
        "48 89 f2 c4 e2 f3 f6 d7 be 3b 00 00 00 c4 e2 c3 f6 c6 "
        "48 29 d0 48 01 cf 48 11 d0 48 89 c2 c4 e2 f3 f6 d6 "
        "48 29 c2 48 01 f9 48 11 c2 48 89 c8 48 83 e8 c5 "
        "48 83 da 00 48 0f 42 c1 c3"
    ),
}

EXPECTED_SEQUENCES = {
    **{
        f"aarch64-{target_os}-{operation}": sequence
        for target_os, target_sequences in AARCH64.items()
        for operation, sequence in target_sequences.items()
    },
    **{f"x86_64-{operation.replace('_', '-')}": sequence for operation, sequence in X86_64.items()},
}

SYMBOL_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
AARCH64_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})(?:\s|$)")
INSTRUCTION_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s+(.*)$")
BYTE_RE = re.compile(r"^[0-9a-fA-F]{2}$")


def objdump(explicit: str | None) -> str:
    if explicit:
        return explicit
    if tool := shutil.which("llvm-objdump"):
        return tool
    if xcrun := shutil.which("xcrun"):
        return subprocess.run(
            [xcrun, "--find", "llvm-objdump"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    raise SystemExit("llvm-objdump was not found")


def disassemble(tool: str, binary: Path, symbol: str) -> str:
    for candidate in (symbol, f"_{symbol}"):
        result = subprocess.run(
            [tool, f"--disassemble-symbols={candidate}", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        if f"<{candidate}>:" in result.stdout:
            return result.stdout
    raise SystemExit(f"symbol {symbol!r} was not found in {binary}")


def parse_words(text: str, symbol: str) -> list[int]:
    current = False
    words: list[int] = []
    for line in text.splitlines():
        if match := SYMBOL_RE.match(line):
            current = match.group(1).removeprefix("_") == symbol
            continue
        if current and (match := AARCH64_RE.match(line)):
            words.append(int(match.group(1), 16))
    return words


def parse_byte_instructions(text: str, symbol: str) -> list[bytes]:
    return [raw for raw, _mnemonic in parse_decoded_byte_instructions(text, symbol)]


def parse_decoded_byte_instructions(text: str, symbol: str) -> list[tuple[bytes, str]]:
    current = False
    result: list[tuple[bytes, str]] = []
    for line in text.splitlines():
        if match := SYMBOL_RE.match(line):
            current = match.group(1).removeprefix("_") == symbol
            continue
        if not current or not (match := INSTRUCTION_RE.match(line)):
            continue
        instruction = bytearray()
        tokens = match.group(1).split()
        byte_count = 0
        for token in tokens:
            if not BYTE_RE.fullmatch(token):
                break
            instruction.append(int(token, 16))
            byte_count += 1
        if instruction:
            result.append((bytes(instruction), " ".join(tokens[byte_count:])))
    return result


def parse_bytes(text: str, symbol: str) -> bytes:
    return b"".join(parse_byte_instructions(text, symbol))


def through_ret(text: str, symbol: str) -> tuple[bytes, list[tuple[bytes, str]]]:
    instructions = parse_decoded_byte_instructions(text, symbol)
    try:
        end = [raw for raw, _mnemonic in instructions].index(b"\xc3") + 1
    except ValueError as error:
        raise SystemExit(f"symbol {symbol!r} has no ret instruction") from error
    return b"".join(raw for raw, _mnemonic in instructions[:end]), instructions[end:]


def darwin_frame(inner: bytes) -> bytes:
    frame_enter = bytes.fromhex("55 48 89 e5")
    frame_leave = bytes.fromhex("5d c3")
    if not inner.endswith(b"\xc3"):
        raise ValueError("proved x86 sequence has no return instruction")
    return frame_enter + inner[:-1] + frame_leave


def require_post_return(
    label: str,
    trailing: list[tuple[bytes, str]],
    policy: str,
) -> None:
    if policy == "none":
        require(f"{label} instructions after ret", trailing, [])
        return
    if policy == "nop-padding-only":
        rejected = [
            (raw, mnemonic)
            for raw, mnemonic in trailing
            if not mnemonic.lower().startswith("nop")
        ]
        require(f"{label} non-NOP instructions after ret", rejected, [])
        return
    if policy == "int3-padding-only":
        rejected = [
            (raw, mnemonic)
            for raw, mnemonic in trailing
            if raw != b"\xcc" or mnemonic.strip().lower() != "int3"
        ]
        require(f"{label} non-INT3 instructions after ret", rejected, [])
        return
    raise SystemExit(f"unknown post-return policy {policy!r}")


def require_format(label: str, disassembly: str, marker: str) -> None:
    if marker not in disassembly:
        raise SystemExit(
            f"{label} object format mismatch\nexpected marker: {marker!r}"
        )


def require(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch\nexpected: {expected!r}\nactual:   {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--add-object", type=Path, required=True)
    parser.add_argument("--sub-object", type=Path, required=True)
    parser.add_argument("--mul-object", type=Path, required=True)
    parser.add_argument("--mul-bmi2-object", type=Path)
    parser.add_argument("--production-witness", type=Path, required=True)
    parser.add_argument("--bmi2-production-witness", type=Path)
    parser.add_argument("--llvm-objdump")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    validate_matrix(matrix)
    try:
        target = target_by_id(matrix, args.target_id)
    except ValueError as error:
        parser.error(str(error))
    architecture = target["architecture"]
    marker = target["object_format_marker"]
    tool = objdump(args.llvm_objdump)

    objects = {"add": args.add_object, "sub": args.sub_object, "mul": args.mul_object}
    if architecture == "aarch64":
        baseline = profile_by_id(target, "baseline")
        for operation, path in objects.items():
            object_symbol = f"jolt_fp64_{operation}_asm"
            witness_symbol = f"jolt_fp64_{operation}_production_witness"
            operation_entry = next(
                entry for entry in baseline["operations"] if entry["name"] == operation
            )
            expected = EXPECTED_SEQUENCES[operation_entry["expected_sequence"]]
            object_disassembly = disassemble(tool, path, object_symbol)
            witness_disassembly = disassemble(tool, args.production_witness, witness_symbol)
            require_format(f"AArch64 {operation} proof object", object_disassembly, marker)
            require_format(f"AArch64 {operation} production witness", witness_disassembly, marker)
            require(
                f"AArch64 {operation} proof object",
                parse_words(object_disassembly, object_symbol),
                expected,
            )
            require(
                f"AArch64 {operation} production witness",
                parse_words(witness_disassembly, witness_symbol),
                expected,
            )
    else:
        baseline = profile_by_id(target, "baseline")
        for operation, path in objects.items():
            object_symbol = f"jolt_fp64_{operation}_asm"
            witness_symbol = f"jolt_fp64_{operation}_production_witness"
            operation_entry = next(
                entry for entry in baseline["operations"] if entry["name"] == operation
            )
            expected = EXPECTED_SEQUENCES[operation_entry["expected_sequence"]]
            object_disassembly = disassemble(tool, path, object_symbol)
            witness_disassembly = disassemble(tool, args.production_witness, witness_symbol)
            require_format(f"x86-64 {operation} proof object", object_disassembly, marker)
            require_format(f"x86-64 {operation} production witness", witness_disassembly, marker)
            require(
                f"x86-64 {operation} proof object",
                parse_bytes(object_disassembly, object_symbol),
                expected,
            )
            actual_witness, trailing = through_ret(witness_disassembly, witness_symbol)
            expected_witness = (
                darwin_frame(expected)
                if target["wrapper_policy"] == "darwin-frame"
                else expected
            )
            require(
                f"x86-64 {operation} production witness",
                actual_witness,
                expected_witness,
            )
            require_post_return(
                f"x86-64 {operation} production witness",
                trailing,
                target["post_return_policy"],
            )
        if args.mul_bmi2_object is None or args.bmi2_production_witness is None:
            parser.error("x86_64 requires both BMI2 paths")
        bmi2 = profile_by_id(target, "bmi2-mul")
        bmi2_operation = bmi2["operations"][0]
        bmi2_expected = EXPECTED_SEQUENCES[bmi2_operation["expected_sequence"]]
        bmi2_object_disassembly = disassemble(
            tool, args.mul_bmi2_object, "jolt_fp64_mul_bmi2_asm"
        )
        bmi2_witness_disassembly = disassemble(
            tool, args.bmi2_production_witness, "jolt_fp64_mul_production_witness"
        )
        require_format(
            "x86-64 BMI2 multiplication proof object",
            bmi2_object_disassembly,
            marker,
        )
        require_format(
            "x86-64 BMI2 multiplication production witness",
            bmi2_witness_disassembly,
            marker,
        )
        require(
            "x86-64 BMI2 multiplication proof object",
            parse_bytes(
                bmi2_object_disassembly,
                "jolt_fp64_mul_bmi2_asm",
            ),
            bmi2_expected,
        )
        bmi2_witness, bmi2_trailing = through_ret(
            bmi2_witness_disassembly,
            "jolt_fp64_mul_production_witness",
        )
        require(
            "x86-64 BMI2 multiplication production witness",
            bmi2_witness,
            darwin_frame(bmi2_expected)
            if target["wrapper_policy"] == "darwin-frame"
            else bmi2_expected,
        )
        require_post_return(
            "x86-64 BMI2 multiplication production witness",
            bmi2_trailing,
            target["post_return_policy"],
        )
    print(
        f"Fp64 matrix entry {target['id']} proof objects and production witnesses match "
        f"({target['certification_scope']})."
    )


if __name__ == "__main__":
    main()
