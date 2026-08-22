#!/usr/bin/env python3
"""Connect proved Fp128 arithmetic objects to optimized public witnesses."""

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
AARCH64_ADD_OBJECT_WORDS = [AARCH64_LOAD_A7F7_INTO_W4, *AARCH64_ADD_BODY, AARCH64_RET]
AARCH64_SUB_OBJECT_WORDS = [AARCH64_LOAD_A7F7_INTO_W4, *AARCH64_SUB_BODY, AARCH64_RET]
AARCH64_MUL_OBJECT_WORDS = [
    AARCH64_LOAD_A7F7_INTO_W4,
    *AARCH64_MUL_BODY,
    AARCH64_RET,
]
AARCH64_ADD_WITNESS_WORDS = [
    AARCH64_LOAD_A7F7_INTO_W4,
    *AARCH64_ADD_BODY,
    AARCH64_RET,
]
AARCH64_SUB_WITNESS_WORDS = [
    AARCH64_LOAD_A7F7_INTO_W4,
    *AARCH64_SUB_BODY,
    AARCH64_RET,
]
AARCH64_MUL_WITNESS_WORDS = [
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
X86_64_RET = bytes([0xC3])
X86_64_LOAD_A7F7_INTO_R8D = bytes.fromhex("41 b8 f7 a7 ff ff")

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


def parse_symbol_bytes(disassembly: str) -> dict[str, bytes]:
    """Return x86 instruction bytes keyed by normalized symbol name."""
    symbols: dict[str, bytearray] = {}
    current: bytearray | None = None
    for line in disassembly.splitlines():
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, bytearray())
            continue
        instruction_match = INSTRUCTION_LINE_RE.match(line)
        if instruction_match is None or current is None:
            continue
        for token in instruction_match.group(1).split():
            if BYTE_TOKEN_RE.fullmatch(token) is None:
                break
            current.append(int(token, 16))
    return {name: bytes(value) for name, value in symbols.items()}


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


def require_bytes_once(label: str, actual: bytes, expected: bytes) -> None:
    occurrences = sum(
        actual.startswith(expected, index)
        for index in range(len(actual) - len(expected) + 1)
    )
    if occurrences != 1:
        raise SystemExit(
            f"{label} expected one exact byte sequence, found {occurrences}\n"
            f"expected sequence: {expected.hex(' ')}\n"
            f"actual symbol:     {actual.hex(' ')}"
        )


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
        AARCH64_ADD_OBJECT_WORDS,
    )
    require_words(
        "public addition witness", add_witness_words, AARCH64_ADD_WITNESS_WORDS
    )
    require_words(
        "standalone subtraction proof object",
        sub_object_words,
        AARCH64_SUB_OBJECT_WORDS,
    )
    require_words(
        "public subtraction witness",
        sub_witness_words,
        AARCH64_SUB_WITNESS_WORDS,
    )
    require_words(
        "standalone multiplication proof object",
        mul_object_words,
        AARCH64_MUL_OBJECT_WORDS,
    )
    require_words(
        "public multiplication witness",
        mul_witness_words,
        AARCH64_MUL_WITNESS_WORDS,
    )


def check_x86_64(
    tool: list[str], add_object: Path, sub_object: Path, witness: Path
) -> None:
    add_object_bytes = read_symbol_bytes(tool, add_object, "jolt_fp128_add_asm")
    sub_object_bytes = read_symbol_bytes(tool, sub_object, "jolt_fp128_sub_asm")
    add_witness_bytes = read_symbol_bytes(
        tool, witness, "jolt_fp128_add_production_witness"
    )
    sub_witness_bytes = read_symbol_bytes(
        tool, witness, "jolt_fp128_sub_production_witness"
    )
    require_bytes(
        "standalone addition proof object",
        add_object_bytes,
        X86_64_LOAD_A7F7_INTO_R8D + X86_64_ADD_BODY + X86_64_RET,
    )
    require_bytes(
        "standalone subtraction proof object",
        sub_object_bytes,
        X86_64_LOAD_A7F7_INTO_R8D + X86_64_SUB_BODY + X86_64_RET,
    )
    require_bytes_once(
        "public addition witness",
        add_witness_bytes,
        X86_64_LOAD_A7F7_INTO_R8D + X86_64_ADD_BODY,
    )
    require_bytes_once(
        "public subtraction witness",
        sub_witness_bytes,
        X86_64_LOAD_A7F7_INTO_R8D + X86_64_SUB_BODY,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=["aarch64", "x86_64"], required=True
    )
    parser.add_argument("--add-object", required=True, type=Path)
    parser.add_argument("--sub-object", required=True, type=Path)
    parser.add_argument("--mul-object", type=Path)
    parser.add_argument("--production-witness", required=True, type=Path)
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
        check_x86_64(tool, args.add_object, args.sub_object, args.production_witness)
    print(f"Fp128 {args.architecture} proof objects and public witnesses match.")


if __name__ == "__main__":
    main()
