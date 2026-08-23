#!/usr/bin/env python3
"""Ratchet proved Fp64 object bytes against compiled production witnesses."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


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
    current = False
    result: list[bytes] = []
    for line in text.splitlines():
        if match := SYMBOL_RE.match(line):
            current = match.group(1).removeprefix("_") == symbol
            continue
        if not current or not (match := INSTRUCTION_RE.match(line)):
            continue
        instruction = bytearray()
        for token in match.group(1).split():
            if not BYTE_RE.fullmatch(token):
                break
            instruction.append(int(token, 16))
        if instruction:
            result.append(bytes(instruction))
    return result


def parse_bytes(text: str, symbol: str) -> bytes:
    return b"".join(parse_byte_instructions(text, symbol))


def through_ret(text: str, symbol: str) -> bytes:
    instructions = parse_byte_instructions(text, symbol)
    try:
        end = instructions.index(b"\xc3") + 1
    except ValueError as error:
        raise SystemExit(f"symbol {symbol!r} has no ret instruction") from error
    return b"".join(instructions[:end])


def normalize_darwin_frame(actual: bytes) -> bytes:
    frame_enter = bytes.fromhex("55 48 89 e5")
    frame_leave = bytes.fromhex("5d c3")
    if actual.startswith(frame_enter) and actual.endswith(frame_leave):
        return actual[len(frame_enter) : -len(frame_leave)] + b"\xc3"
    return actual


def require(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch\nexpected: {expected!r}\nactual:   {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("aarch64", "x86_64"), required=True)
    parser.add_argument("--target-os", choices=("darwin", "linux"), required=True)
    parser.add_argument("--add-object", type=Path, required=True)
    parser.add_argument("--sub-object", type=Path, required=True)
    parser.add_argument("--mul-object", type=Path, required=True)
    parser.add_argument("--mul-bmi2-object", type=Path)
    parser.add_argument("--production-witness", type=Path, required=True)
    parser.add_argument("--bmi2-production-witness", type=Path)
    parser.add_argument("--llvm-objdump")
    args = parser.parse_args()
    tool = objdump(args.llvm_objdump)

    objects = {"add": args.add_object, "sub": args.sub_object, "mul": args.mul_object}
    if args.architecture == "aarch64":
        expected_sequences = AARCH64[args.target_os]
        for operation, path in objects.items():
            object_symbol = f"jolt_fp64_{operation}_asm"
            witness_symbol = f"jolt_fp64_{operation}_production_witness"
            expected = expected_sequences[operation]
            require(
                f"AArch64 {operation} proof object",
                parse_words(disassemble(tool, path, object_symbol), object_symbol),
                expected,
            )
            require(
                f"AArch64 {operation} production witness",
                parse_words(disassemble(tool, args.production_witness, witness_symbol), witness_symbol),
                expected,
            )
    else:
        for operation, path in objects.items():
            object_symbol = f"jolt_fp64_{operation}_asm"
            witness_symbol = f"jolt_fp64_{operation}_production_witness"
            expected = X86_64[operation]
            require(
                f"x86-64 {operation} proof object",
                parse_bytes(disassemble(tool, path, object_symbol), object_symbol),
                expected,
            )
            require(
                f"x86-64 {operation} production witness",
                normalize_darwin_frame(
                    through_ret(disassemble(tool, args.production_witness, witness_symbol), witness_symbol)
                ),
                expected,
            )
        if args.mul_bmi2_object is None or args.bmi2_production_witness is None:
            parser.error("x86_64 requires both BMI2 paths")
        require(
            "x86-64 BMI2 multiplication proof object",
            parse_bytes(
                disassemble(tool, args.mul_bmi2_object, "jolt_fp64_mul_bmi2_asm"),
                "jolt_fp64_mul_bmi2_asm",
            ),
            X86_64["mul_bmi2"],
        )
        require(
            "x86-64 BMI2 multiplication production witness",
            normalize_darwin_frame(through_ret(
                disassemble(
                    tool,
                    args.bmi2_production_witness,
                    "jolt_fp64_mul_production_witness",
                ),
                "jolt_fp64_mul_production_witness",
            )),
            X86_64["mul_bmi2"],
        )
    print(f"Fp64 {args.architecture} proof objects and production witnesses match.")


if __name__ == "__main__":
    main()
