#!/usr/bin/env python3
"""Connect proved Fp128 arithmetic objects to compiled inspection witnesses."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


AARCH64_ADD_BODY = [
    0xAB020005,
    0xBA030026,
    0x1A9F37E7,
    0xAB0400A8,
    0xBA1F00C9,
    0x7A4038E0,
    0x9A851100,
    0x9A861121,
]
AARCH64_SUB_BODY = [
    0xEB020005,
    0xFA030026,
    0x9A8423E7,
    0xEB0700A0,
    0xDA1F00C1,
]
AARCH64_MUL_BODY = [
    0x9B027C05,
    0x9BC27C06,
    0x9B037C07,
    0x9BC37C08,
    0x9B027C29,
    0x9BC27C2A,
    0x9B037C2B,
    0x9BC37C2C,
    0xAB0700C6,
    0x1A9F37E7,
    0xAB0A0108,
    0x1A9F37EA,
    0xAB0B0108,
    0x9A8A354A,
    0xAB0900C6,
    0xBA070108,
    0x9A0A018C,
    0x9B047D07,
    0x9BC47D09,
    0x9B047D8A,
    0x9BC47D8B,
    0xAB0700A5,
    0xBA0900C6,
    0x1A9F37E8,
    0xAB0A00C6,
    0x9A08016C,
    0x9B047D87,
    0xAB0700A5,
    0xBA1F00C6,
    0x1A9F37E7,
    0xAB0400A9,
    0xBA1F00CA,
    0x7A4038E0,
    0x9A851120,
    0x9A861141,
]
AARCH64_RET = 0xD65F03C0
AARCH64_LOAD_A7F7_INTO_W4 = 0x128B0104
AARCH64_ADD_WORDS = [AARCH64_LOAD_A7F7_INTO_W4, *AARCH64_ADD_BODY, AARCH64_RET]
AARCH64_SUB_WORDS = [AARCH64_LOAD_A7F7_INTO_W4, *AARCH64_SUB_BODY, AARCH64_RET]
AARCH64_MUL_WORDS = [
    AARCH64_LOAD_A7F7_INTO_W4,
    *AARCH64_MUL_BODY,
    AARCH64_RET,
]

X86_64_ADD_BODY = bytes.fromhex(
    "48 01 d7 48 11 ce 4d 19 c9 49 89 fa 49 89 f3 4d 01 c2 "
    "49 83 d3 00 4d 11 c9 49 0f 45 fa 49 0f 45 f3"
)
X86_64_SUB_BODY = bytes.fromhex(
    "48 29 d7 48 19 ce 4d 19 c9 4d 21 c1 4c 29 cf 48 83 de 00"
)
X86_64_MUL_BODY = bytes.fromhex(
    "49 89 d2 49 89 cb 49 89 f9 48 89 f8 49 f7 e2 48 89 c7 "
    "48 89 d1 4c 89 c8 49 f7 e3 48 01 c1 48 83 d2 00 49 89 "
    "d1 48 89 f0 49 f7 e2 45 31 d2 48 01 c1 49 11 d1 49 83 "
    "d2 00 48 89 f0 49 f7 e3 49 01 c1 4c 11 d2 49 89 d3 4c "
    "89 c8 49 f7 e0 45 31 d2 48 01 c7 48 11 d1 49 83 d2 00 "
    "4c 89 d8 49 f7 e0 48 01 c1 4c 11 d2 48 89 d0 49 f7 e0 "
    "45 31 c9 48 01 c7 48 83 d1 00 49 83 d1 00 48 89 f8 48 "
    "89 ca 4c 01 c0 48 83 d2 00 4d 11 c9 48 0f 45 f8 48 0f "
    "45 ca"
)
X86_64_MUL_BMI2_ADX_BODY = bytes.fromhex(
    "c4 62 b3 f6 d7 c4 62 fb f6 de 48 89 ca c4 e2 f3 f6 ff "
    "c4 62 cb f6 c6 31 d2 66 4c 0f 38 f6 d0 f3 4c 0f 38 f6 d1 "
    "66 4c 0f 38 f6 df f3 4c 0f 38 f6 de 66 4c 0f 38 f6 c2 "
    "f3 4c 0f 38 f6 c2 ba f7 a7 ff ff c4 c2 cb f6 fb c4 c2 fb f6 c8 "
    "49 01 f1 49 11 fa 48 83 d1 00 49 01 c2 48 83 d1 00 48 0f af ca "
    "49 01 c9 49 83 d2 00 4d 19 db 4c 89 c8 48 01 d0 4c 89 d2 "
    "48 83 d2 00 4d 11 db 49 0f 44 c1 49 0f 44 d2"
)
X86_64_RET = bytes([0xC3])
X86_64_LOAD_A7F7_INTO_R8D = bytes.fromhex("41 b8 f7 a7 ff ff")
X86_64_ADD_SUB_ABI_RETURN = bytes.fromhex("48 89 f2 48 89 f8")
X86_64_MUL_ABI_RETURN = bytes.fromhex("48 89 f8 48 89 ca")
X86_64_DARWIN_FRAME_ENTER = bytes.fromhex("55 48 89 e5")
X86_64_DARWIN_FRAME_LEAVE = bytes.fromhex("5d c3")

SYMBOL_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
AARCH64_INSTRUCTION_RE = re.compile(
    r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})(?:\s|$)"
)
INSTRUCTION_LINE_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s+(.*)$")
BYTE_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{2}$")


def parse_symbol_words(disassembly: str) -> dict[str, list[int]]:
    """Return AArch64 instruction words keyed by normalized symbol name."""
    symbols: dict[str, list[int]] = {}
    current: list[int] | None = None
    for line in disassembly.splitlines():
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, [])
            continue
        instruction_match = AARCH64_INSTRUCTION_RE.match(line)
        if instruction_match and current is not None:
            current.append(int(instruction_match.group(1), 16))
    return symbols


def parse_symbol_byte_instructions(disassembly: str) -> dict[str, list[bytes]]:
    """Return decoded x86 instruction byte strings by normalized symbol."""
    symbols: dict[str, list[bytes]] = {}
    current: list[bytes] | None = None
    for line in disassembly.splitlines():
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, [])
            continue
        instruction_match = INSTRUCTION_LINE_RE.match(line)
        if instruction_match is None or current is None:
            continue
        instruction = bytearray()
        for token in instruction_match.group(1).split():
            if BYTE_TOKEN_RE.fullmatch(token) is None:
                break
            instruction.append(int(token, 16))
        if instruction:
            current.append(bytes(instruction))
    return symbols


def parse_symbol_bytes(disassembly: str) -> dict[str, bytes]:
    """Return flattened x86 instruction bytes keyed by normalized symbol."""
    return {
        name: b"".join(instructions)
        for name, instructions in parse_symbol_byte_instructions(disassembly).items()
    }


def find_llvm_objdump(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    llvm_objdump = shutil.which("llvm-objdump")
    if llvm_objdump:
        return [llvm_objdump]
    xcrun = shutil.which("xcrun")
    if xcrun:
        result = subprocess.run(
            [xcrun, "--find", "llvm-objdump"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [result.stdout.strip()]
    raise SystemExit("llvm-objdump was not found")


def read_symbol_disassembly(tool: list[str], binary: Path, symbol: str) -> str:
    for candidate in (symbol, f"_{symbol}"):
        result = subprocess.run(
            [*tool, f"--disassemble-symbols={candidate}", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        if f"<{candidate}>:" in result.stdout:
            return result.stdout
    raise SystemExit(f"symbol {symbol!r} was not found in {binary}")


def read_symbol_words(tool: list[str], binary: Path, symbol: str) -> list[int]:
    disassembly = read_symbol_disassembly(tool, binary, symbol)
    words = parse_symbol_words(disassembly).get(symbol)
    if words is None:
        raise SystemExit(f"symbol {symbol!r} had no AArch64 instructions in {binary}")
    return words


def read_symbol_bytes(tool: list[str], binary: Path, symbol: str) -> bytes:
    disassembly = read_symbol_disassembly(tool, binary, symbol)
    value = parse_symbol_bytes(disassembly).get(symbol)
    if value is None:
        raise SystemExit(f"symbol {symbol!r} had no x86-64 instructions in {binary}")
    return value


def read_symbol_byte_instructions(
    tool: list[str], binary: Path, symbol: str
) -> list[bytes]:
    disassembly = read_symbol_disassembly(tool, binary, symbol)
    value = parse_symbol_byte_instructions(disassembly).get(symbol)
    if value is None:
        raise SystemExit(f"symbol {symbol!r} had no x86-64 instructions in {binary}")
    return value


def format_words(words: list[int]) -> str:
    return " ".join(f"{word:08x}" for word in words)


def require_words(label: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise SystemExit(
            f"{label} instruction mismatch\n"
            f"expected: {format_words(expected)}\n"
            f"actual:   {format_words(actual)}"
        )


def require_bytes(label: str, actual: bytes, expected: bytes) -> None:
    if actual != expected:
        raise SystemExit(
            f"{label} byte mismatch\n"
            f"expected: {expected.hex(' ')}\n"
            f"actual:   {actual.hex(' ')}"
        )


def instructions_through_ret(label: str, instructions: list[bytes]) -> bytes:
    """Return one function body through its first decoded near return."""
    for index, instruction in enumerate(instructions):
        if instruction == X86_64_RET:
            return b"".join(instructions[: index + 1])
    raise SystemExit(f"{label} had no decoded ret instruction")


def check_aarch64(
    tool: list[str],
    add_object: Path,
    sub_object: Path,
    mul_object: Path,
    witness: Path,
) -> None:
    add_object_words = read_symbol_words(tool, add_object, "jolt_fp128_add_asm")
    sub_object_words = read_symbol_words(tool, sub_object, "jolt_fp128_sub_asm")
    mul_object_words = read_symbol_words(tool, mul_object, "jolt_fp128_mul_asm")
    add_witness_words = read_symbol_words(
        tool, witness, "jolt_fp128_add_production_witness"
    )
    sub_witness_words = read_symbol_words(
        tool, witness, "jolt_fp128_sub_production_witness"
    )
    mul_witness_words = read_symbol_words(
        tool, witness, "jolt_fp128_mul_production_witness"
    )
    require_words(
        "standalone addition proof object",
        add_object_words,
        AARCH64_ADD_WORDS,
    )
    require_words("addition inspection witness", add_witness_words, AARCH64_ADD_WORDS)
    require_words(
        "standalone subtraction proof object",
        sub_object_words,
        AARCH64_SUB_WORDS,
    )
    require_words(
        "subtraction inspection witness",
        sub_witness_words,
        AARCH64_SUB_WORDS,
    )
    require_words(
        "standalone multiplication proof object",
        mul_object_words,
        AARCH64_MUL_WORDS,
    )
    require_words(
        "multiplication inspection witness",
        mul_witness_words,
        AARCH64_MUL_WORDS,
    )


def check_x86_64(
    tool: list[str],
    add_object: Path,
    sub_object: Path,
    mul_object: Path,
    witness: Path,
    bmi2_adx_mul_object: Path,
    bmi2_adx_witness: Path,
    target_os: str,
) -> None:
    add_object_bytes = read_symbol_bytes(tool, add_object, "jolt_fp128_add_asm")
    sub_object_bytes = read_symbol_bytes(tool, sub_object, "jolt_fp128_sub_asm")
    mul_object_bytes = read_symbol_bytes(tool, mul_object, "jolt_fp128_mul_asm")
    bmi2_adx_mul_object_bytes = read_symbol_bytes(
        tool, bmi2_adx_mul_object, "jolt_fp128_mul_bmi2_adx_asm"
    )
    add_witness_instructions = read_symbol_byte_instructions(
        tool, witness, "jolt_fp128_add_production_witness"
    )
    sub_witness_instructions = read_symbol_byte_instructions(
        tool, witness, "jolt_fp128_sub_production_witness"
    )
    mul_witness_instructions = read_symbol_byte_instructions(
        tool, witness, "jolt_fp128_mul_production_witness"
    )
    bmi2_adx_witness_instructions = read_symbol_byte_instructions(
        tool, bmi2_adx_witness, "jolt_fp128_mul_production_witness"
    )
    add_expected = (
        X86_64_LOAD_A7F7_INTO_R8D
        + X86_64_ADD_BODY
        + X86_64_ADD_SUB_ABI_RETURN
        + X86_64_RET
    )
    sub_expected = (
        X86_64_LOAD_A7F7_INTO_R8D
        + X86_64_SUB_BODY
        + X86_64_ADD_SUB_ABI_RETURN
        + X86_64_RET
    )
    mul_expected = (
        X86_64_LOAD_A7F7_INTO_R8D
        + X86_64_MUL_BODY
        + X86_64_MUL_ABI_RETURN
        + X86_64_RET
    )
    bmi2_adx_mul_expected = X86_64_MUL_BMI2_ADX_BODY + X86_64_RET
    add_darwin_expected = (
        X86_64_DARWIN_FRAME_ENTER
        + add_expected[:-1]
        + X86_64_DARWIN_FRAME_LEAVE
    )
    sub_darwin_expected = (
        X86_64_DARWIN_FRAME_ENTER
        + sub_expected[:-1]
        + X86_64_DARWIN_FRAME_LEAVE
    )
    mul_darwin_expected = (
        X86_64_DARWIN_FRAME_ENTER
        + mul_expected[:-1]
        + X86_64_DARWIN_FRAME_LEAVE
    )
    bmi2_adx_mul_darwin_expected = (
        X86_64_DARWIN_FRAME_ENTER
        + X86_64_MUL_BMI2_ADX_BODY
        + X86_64_DARWIN_FRAME_LEAVE
    )
    require_bytes(
        "standalone addition proof object",
        add_object_bytes,
        add_expected,
    )
    require_bytes(
        "standalone subtraction proof object",
        sub_object_bytes,
        sub_expected,
    )
    require_bytes(
        "standalone multiplication proof object",
        mul_object_bytes,
        mul_expected,
    )
    require_bytes(
        "standalone BMI2 and ADX multiplication proof object",
        bmi2_adx_mul_object_bytes,
        bmi2_adx_mul_expected,
    )
    add_witness_expected = (
        add_darwin_expected if target_os == "darwin" else add_expected
    )
    sub_witness_expected = (
        sub_darwin_expected if target_os == "darwin" else sub_expected
    )
    mul_witness_expected = (
        mul_darwin_expected if target_os == "darwin" else mul_expected
    )
    require_bytes(
        "addition inspection witness",
        instructions_through_ret(
            "addition inspection witness", add_witness_instructions
        ),
        add_witness_expected,
    )
    require_bytes(
        "subtraction inspection witness",
        instructions_through_ret(
            "subtraction inspection witness", sub_witness_instructions
        ),
        sub_witness_expected,
    )
    require_bytes(
        "multiplication inspection witness",
        instructions_through_ret(
            "multiplication inspection witness", mul_witness_instructions
        ),
        mul_witness_expected,
    )
    bmi2_adx_witness_expected = (
        bmi2_adx_mul_darwin_expected
        if target_os == "darwin"
        else bmi2_adx_mul_expected
    )
    require_bytes(
        "BMI2 and ADX multiplication inspection witness",
        instructions_through_ret(
            "BMI2 and ADX multiplication inspection witness",
            bmi2_adx_witness_instructions,
        ),
        bmi2_adx_witness_expected,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=["aarch64", "x86_64"], required=True
    )
    parser.add_argument("--target-os", choices=["darwin", "linux"], required=True)
    parser.add_argument("--add-object", required=True, type=Path)
    parser.add_argument("--sub-object", required=True, type=Path)
    parser.add_argument("--mul-object", type=Path)
    parser.add_argument("--bmi2-adx-mul-object", type=Path)
    parser.add_argument("--production-witness", required=True, type=Path)
    parser.add_argument("--bmi2-adx-production-witness", type=Path)
    parser.add_argument("--llvm-objdump")
    args = parser.parse_args()

    tool = find_llvm_objdump(args.llvm_objdump)
    if args.architecture == "aarch64":
        if args.mul_object is None:
            parser.error("--mul-object is required for aarch64")
        check_aarch64(
            tool,
            args.add_object,
            args.sub_object,
            args.mul_object,
            args.production_witness,
        )
    else:
        if args.mul_object is None:
            parser.error("--mul-object is required for x86_64")
        if args.bmi2_adx_mul_object is None:
            parser.error("--bmi2-adx-mul-object is required for x86_64")
        if args.bmi2_adx_production_witness is None:
            parser.error("--bmi2-adx-production-witness is required for x86_64")
        check_x86_64(
            tool,
            args.add_object,
            args.sub_object,
            args.mul_object,
            args.production_witness,
            args.bmi2_adx_mul_object,
            args.bmi2_adx_production_witness,
            args.target_os,
        )
    print(f"Fp128 {args.architecture} proof objects and inspection witnesses match.")


if __name__ == "__main__":
    main()
