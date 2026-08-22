#!/usr/bin/env python3
"""Check that the proved and public Jolt Fp128 add/sub words agree."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ADD_BODY = [
    0xAB020005,
    0xBA030026,
    0x1A9F37E7,
    0xAB0400A8,
    0xBA1F00C9,
    0x7A4038E0,
    0x9A851100,
    0x9A861121,
]
SUB_BODY = [
    0xEB020005,
    0xFA030026,
    0x9A8423E7,
    0xEB0700A0,
    0xDA1F00C1,
]
RET = 0xD65F03C0
LOAD_A7F7_INTO_W4 = 0x128B0104

ADD_OBJECT_WORDS = [*ADD_BODY, RET]
SUB_OBJECT_WORDS = [*SUB_BODY, RET]
ADD_PRODUCTION_WITNESS_WORDS = [LOAD_A7F7_INTO_W4, *ADD_BODY, RET]
SUB_PRODUCTION_WITNESS_WORDS = [LOAD_A7F7_INTO_W4, *SUB_BODY, RET]

SYMBOL_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
INSTRUCTION_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})(?:\s|$)")


def parse_symbol_words(disassembly: str) -> dict[str, list[int]]:
    """Return instruction words keyed by symbol, without a Mach-O underscore."""
    symbols: dict[str, list[int]] = {}
    current: list[int] | None = None
    for line in disassembly.splitlines():
        symbol_match = SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, [])
            continue
        instruction_match = INSTRUCTION_RE.match(line)
        if instruction_match and current is not None:
            current.append(int(instruction_match.group(1), 16))
    return symbols


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


def read_symbol_words(tool: list[str], binary: Path, symbol: str) -> list[int]:
    for candidate in (symbol, f"_{symbol}"):
        result = subprocess.run(
            [*tool, f"--disassemble-symbols={candidate}", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        words = parse_symbol_words(result.stdout).get(symbol)
        if words is not None:
            return words
    raise SystemExit(f"symbol {symbol!r} was not found in {binary}")


def format_words(words: list[int]) -> str:
    return " ".join(f"{word:08x}" for word in words)


def require_words(label: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise SystemExit(
            f"{label} instruction mismatch\n"
            f"expected: {format_words(expected)}\n"
            f"actual:   {format_words(actual)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-object", required=True, type=Path)
    parser.add_argument("--sub-object", required=True, type=Path)
    parser.add_argument("--production-witness", required=True, type=Path)
    parser.add_argument("--llvm-objdump")
    args = parser.parse_args()

    tool = find_llvm_objdump(args.llvm_objdump)
    add_object_words = read_symbol_words(tool, args.add_object, "jolt_fp128_add_asm")
    sub_object_words = read_symbol_words(tool, args.sub_object, "jolt_fp128_sub_asm")
    add_witness_words = read_symbol_words(
        tool,
        args.production_witness,
        "jolt_fp128_add_production_witness",
    )
    sub_witness_words = read_symbol_words(
        tool,
        args.production_witness,
        "jolt_fp128_sub_production_witness",
    )
    require_words("standalone addition proof object", add_object_words, ADD_OBJECT_WORDS)
    require_words(
        "public addition witness",
        add_witness_words,
        ADD_PRODUCTION_WITNESS_WORDS,
    )
    require_words("standalone subtraction proof object", sub_object_words, SUB_OBJECT_WORDS)
    require_words(
        "public subtraction witness",
        sub_witness_words,
        SUB_PRODUCTION_WITNESS_WORDS,
    )
    print("Fp128 add/sub proof objects and public witness bytes match.")


if __name__ == "__main__":
    main()
