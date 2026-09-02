#!/usr/bin/env python3
"""Find proved field instruction sequences in a linked executable.

Exact matches inherit the existing byte-level linkage. Structural matches are
only candidate proof instances: they have the same decoded instruction
template under one consistent, injective register renaming, but still require
a HOL Light renaming theorem and a check of any value prepared outside the
matched body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_fp128_proof_artifacts as fp128  # noqa: E402
import check_fp64_proof_artifacts as fp64  # noqa: E402


SYMBOL_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
BYTE_RE = re.compile(r"^[0-9a-fA-F]{2}$")
WORD_RE = re.compile(r"^([0-9a-fA-F]{8})(?:\s|$)")


@dataclass(frozen=True)
class Instruction:
    address: int
    encoding: bytes
    text: str


@dataclass(frozen=True)
class Match:
    family: str
    operation: str
    symbol: str
    address: str


@dataclass(frozen=True)
class StructuralMatch:
    family: str
    operation: str
    symbol: str
    address: str
    register_mapping: dict[str, str]
    external_inputs: tuple[str, ...]


AARCH64_REGISTER_RE = re.compile(r"\b([xw])([0-9]|[12][0-9]|30|zr)\b", re.I)
X86_REGISTER_RE = re.compile(
    r"%(r(?:1[0-5]|[89])(?:d|w|b)?|"
    r"r(?:ax|bx|cx|dx|si|di|bp|sp)|"
    r"e(?:ax|bx|cx|dx|si|di|bp|sp)|"
    r"(?:ax|bx|cx|dx|si|di|bp|sp)|"
    r"(?:al|bl|cl|dl|sil|dil|bpl|spl))\b",
    re.I,
)


def expected_patterns(architecture: str) -> dict[str, dict[str, bytes]]:
    if architecture == "aarch64":
        word_bytes = lambda words: b"".join(
            word.to_bytes(4, "little") for word in words
        )
        return {
            "fp128": {
                "add": word_bytes(
                    [fp128.AARCH64_LOAD_A7F7_INTO_W4, *fp128.AARCH64_ADD_BODY]
                ),
                "sub": word_bytes(
                    [fp128.AARCH64_LOAD_A7F7_INTO_W4, *fp128.AARCH64_SUB_BODY]
                ),
                "mul": word_bytes(
                    [fp128.AARCH64_LOAD_A7F7_INTO_W4, *fp128.AARCH64_MUL_BODY]
                ),
            },
            "fp64": {
                f"{operation}_{target_os}": word_bytes(words[:-1])
                for target_os, operations in fp64.AARCH64.items()
                for operation, words in operations.items()
            },
        }
    return {
        "fp128": {
            "add": fp128.X86_64_LOAD_A7F7_INTO_R8D + fp128.X86_64_ADD_BODY,
            "sub": fp128.X86_64_LOAD_A7F7_INTO_R8D + fp128.X86_64_SUB_BODY,
            "mul_baseline": (
                fp128.X86_64_LOAD_A7F7_INTO_R8D + fp128.X86_64_MUL_BODY
            ),
            "mul_bmi2_adx": fp128.X86_64_MUL_BMI2_ADX_BODY,
        },
        "fp64": {
            operation: encoding[:-1]
            for operation, encoding in fp64.X86_64.items()
        },
    }


def parse_instruction_line(line: str, architecture: str) -> Instruction | None:
    if (address_match := ADDRESS_RE.match(line)) is None:
        return None
    address = int(address_match.group(1), 16)
    remainder = address_match.group(2)
    if architecture == "aarch64":
        if (word_match := WORD_RE.match(remainder)) is None:
            return None
        encoding = int(word_match.group(1), 16).to_bytes(4, "little")
        instruction_text = remainder[word_match.end() :]
    else:
        tokens = remainder.split()
        raw = bytearray()
        consumed = 0
        for token in tokens:
            if BYTE_RE.fullmatch(token) is None:
                break
            raw.append(int(token, 16))
            consumed += 1
        if not raw:
            return None
        encoding = bytes(raw)
        instruction_text = " ".join(tokens[consumed:])
    instruction_text = instruction_text.split(";", 1)[0].strip().lower()
    return Instruction(address, encoding, instruction_text)


def parse_disassembly(text: str, architecture: str) -> dict[str, list[Instruction]]:
    symbols: dict[str, list[Instruction]] = {}
    current: list[Instruction] | None = None
    for line in text.splitlines():
        if symbol_match := SYMBOL_RE.match(line):
            name = symbol_match.group(1).removeprefix("_")
            current = symbols.setdefault(name, [])
            continue
        if current is None:
            continue
        if instruction := parse_instruction_line(line, architecture):
            append_instruction(current, instruction, architecture)
    return symbols


def append_instruction(
    instructions: list[Instruction], instruction: Instruction, architecture: str
) -> None:
    if (
        architecture == "x86_64"
        and instructions
        and (not instruction.text or not instructions[-1].text)
    ):
        previous = instructions[-1]
        instructions[-1] = Instruction(
            previous.address,
            previous.encoding + instruction.encoding,
            previous.text or instruction.text,
        )
    else:
        instructions.append(instruction)


def iter_disassembly_symbols(
    tool: list[str], binary: Path, architecture: str
) -> Iterator[tuple[str, list[Instruction]]]:
    """Stream symbols from llvm-objdump so large linked binaries stay bounded."""
    process = subprocess.Popen(
        [*tool, "--disassemble", "--demangle", str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    current_name: str | None = None
    current: list[Instruction] = []
    for line in process.stdout:
        if symbol_match := SYMBOL_RE.match(line):
            if current_name is not None:
                yield current_name, current
            current_name = symbol_match.group(1).removeprefix("_")
            current = []
        elif current_name is not None:
            if instruction := parse_instruction_line(line, architecture):
                append_instruction(current, instruction, architecture)
    if current_name is not None:
        yield current_name, current
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.wait() != 0:
        raise SystemExit(f"llvm-objdump failed:\n{stderr.rstrip()}")


def find_matches_in_symbols(
    symbols: Iterable[tuple[str, list[Instruction]]],
    patterns: dict[str, dict[str, bytes]],
) -> list[Match]:
    matches: list[Match] = []
    by_first_byte: dict[int, list[tuple[str, str, bytes]]] = {}
    for family, operations in patterns.items():
        for operation, expected in operations.items():
            by_first_byte.setdefault(expected[0], []).append(
                (family, operation, expected)
            )
    for symbol, instructions in symbols:
        for start, instruction in enumerate(instructions):
            candidates = by_first_byte.get(instruction.encoding[0], [])
            for family, operation, expected in candidates:
                if not expected.startswith(instruction.encoding):
                    continue
                actual = bytearray()
                for candidate in instructions[start:]:
                    actual.extend(candidate.encoding)
                    if len(actual) >= len(expected):
                        break
                if bytes(actual) == expected:
                    matches.append(
                        Match(
                            family=family,
                            operation=operation,
                            symbol=symbol,
                            address=f"0x{instruction.address:x}",
                        )
                    )
    return matches


def find_matches(
    symbols: dict[str, list[Instruction]],
    patterns: dict[str, dict[str, bytes]],
) -> list[Match]:
    return find_matches_in_symbols(symbols.items(), patterns)


def x86_register(register: str) -> tuple[str, str]:
    name = register.lower()
    if numbered := re.fullmatch(r"(r(?:1[0-5]|[89]))(d|w|b)?", name):
        family, suffix = numbered.groups()
        suffix = suffix or ""
        return family, {"": "64", "d": "32", "w": "16", "b": "8"}[suffix]
    aliases = {
        "rax": ("rax", "64"),
        "eax": ("rax", "32"),
        "ax": ("rax", "16"),
        "al": ("rax", "8"),
        "rbx": ("rbx", "64"),
        "ebx": ("rbx", "32"),
        "bx": ("rbx", "16"),
        "bl": ("rbx", "8"),
        "rcx": ("rcx", "64"),
        "ecx": ("rcx", "32"),
        "cx": ("rcx", "16"),
        "cl": ("rcx", "8"),
        "rdx": ("rdx", "64"),
        "edx": ("rdx", "32"),
        "dx": ("rdx", "16"),
        "dl": ("rdx", "8"),
    }
    for family, short in (("rsi", "si"), ("rdi", "di"), ("rbp", "bp"), ("rsp", "sp")):
        aliases |= {
            family: (family, "64"), f"e{short}": (family, "32"),
            short: (family, "16"), f"{short}l": (family, "8"),
        }
    return aliases[name]


def register_parts(text: str, architecture: str) -> tuple[str, list[tuple[str, str]]]:
    registers: list[tuple[str, str]] = []
    regex = AARCH64_REGISTER_RE if architecture == "aarch64" else X86_REGISTER_RE

    def replace(match: re.Match[str]) -> str:
        if architecture == "aarch64":
            width, number = match.groups()
            family = "zr" if number.lower() == "zr" else f"x{number}"
            registers.append((family, width.lower()))
        else:
            registers.append(x86_register(match.group(1)))
        return "<reg>"

    skeleton = regex.sub(replace, " ".join(text.split()))
    return skeleton, registers


def unify_instruction_templates(
    expected: list[Instruction], actual: list[Instruction], architecture: str
) -> dict[str, str] | None:
    if len(expected) != len(actual):
        return None
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for expected_instruction, actual_instruction in zip(expected, actual, strict=True):
        expected_skeleton, expected_registers = register_parts(
            expected_instruction.text, architecture
        )
        actual_skeleton, actual_registers = register_parts(
            actual_instruction.text, architecture
        )
        if (
            expected_skeleton != actual_skeleton
            or len(expected_registers) != len(actual_registers)
        ):
            return None
        for (expected_family, expected_width), (actual_family, actual_width) in zip(
            expected_registers, actual_registers, strict=True
        ):
            if expected_width != actual_width:
                return None
            if expected_family == "zr" or actual_family == "zr":
                if expected_family != actual_family:
                    return None
                continue
            if forward.get(expected_family, actual_family) != actual_family:
                return None
            if reverse.get(actual_family, expected_family) != expected_family:
                return None
            forward[expected_family] = actual_family
            reverse[actual_family] = expected_family
    return forward


def find_structural_matches_in_symbols(
    symbols: Iterable[tuple[str, list[Instruction]]],
    templates: dict[str, dict[str, list[Instruction]]],
    architecture: str,
) -> list[StructuralMatch]:
    matches: list[StructuralMatch] = []
    for symbol, instructions in symbols:
        for family, operations in templates.items():
            for operation, template in operations.items():
                for start in range(0, len(instructions) - len(template) + 1):
                    candidate = instructions[start : start + len(template)]
                    if not candidate[0].text or not template[0].text:
                        continue
                    if (
                        candidate[0].text.split(maxsplit=1)[0]
                        != template[0].text.split(maxsplit=1)[0]
                    ):
                        continue
                    mapping = unify_instruction_templates(
                        template, candidate, architecture
                    )
                    if mapping is None:
                        continue
                    # The fixed proof objects prepare the Solinas constant before
                    # the body. Inline production code may prepare it elsewhere.
                    constant = "x4" if architecture == "aarch64" else "r8"
                    needs_external_constant = family == "fp128" and operation in {
                        "add",
                        "sub",
                        "mul",
                        "mul_baseline",
                    }
                    external = (
                        (mapping[constant],)
                        if needs_external_constant and constant in mapping
                        else ()
                    )
                    matches.append(
                        StructuralMatch(
                            family, operation, symbol,
                            f"0x{candidate[0].address:x}", mapping, external,
                        )
                    )
    return matches


def validate_required(family: str, matches: list[Match]) -> None:
    operations = {
        match.operation for match in matches if match.family == family
    }
    missing = [
        operation
        for operation in ("add", "sub")
        if not any(candidate.startswith(operation) for candidate in operations)
    ]
    if not any(operation.startswith("mul") for operation in operations):
        missing.append("mul")
    if missing:
        raise SystemExit(
            f"required {family} final-binary sequences were absent: "
            f"{', '.join(missing)}"
        )


def run_objdump(tool: list[str], binary: Path) -> str:
    result = subprocess.run(
        [*tool, "--disassemble", "--demangle", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_proof_object_argument(value: str) -> tuple[str, str, Path]:
    try:
        label, raw_path = value.split("=", 1)
        family, operation = label.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected FAMILY:OPERATION=PATH"
        ) from error
    if family not in {"fp128", "fp64"} or not operation:
        raise argparse.ArgumentTypeError("family must be fp128 or fp64")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"proof object does not exist: {path}")
    return family, operation, path


def load_structural_templates(
    tool: list[str],
    architecture: str,
    objects: list[tuple[str, str, Path]],
) -> dict[str, dict[str, list[Instruction]]]:
    templates: dict[str, dict[str, list[Instruction]]] = {}
    for family, operation, path in objects:
        symbols = parse_disassembly(run_objdump(tool, path), architecture)
        instructions = next((value for value in symbols.values() if value), None)
        if instructions is None or len(instructions) < 2:
            raise SystemExit(f"proof object has no complete instruction body: {path}")
        body = instructions[:-1]
        if family == "fp128" and operation in {
            "add",
            "sub",
            "mul",
            "mul_baseline",
        }:
            body = body[1:]
        templates.setdefault(family, {})[operation] = body
    return templates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def report(
    binary: Path,
    architecture: str,
    matches: list[Match],
    structural_matches: list[StructuralMatch],
    families: list[str],
    verbose_candidates: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "binary": str(binary.resolve()),
        "sha256": sha256_file(binary),
        "architecture": architecture,
        "exact_matches": [asdict(match) for match in matches],
        "structural_candidate_matches": [
            asdict(match) for match in structural_matches
        ],
    }
    print(f"Final binary: {result['binary']}")
    print(f"SHA-256: {result['sha256']}")
    for family in families:
        family_matches = [match for match in matches if match.family == family]
        print(f"{family}: {len(family_matches)} exact instruction sequence matches")
        for match in family_matches:
            print(f"  {match.operation} {match.address} <{match.symbol}>")
        candidates = [
            match for match in structural_matches if match.family == family
        ]
        print(f"{family}: {len(candidates)} structural candidate proof instances")
        operation_counts: dict[str, int] = {}
        for match in candidates:
            operation_counts[match.operation] = (
                operation_counts.get(match.operation, 0) + 1
            )
        for operation, count in sorted(operation_counts.items()):
            print(f"  {operation}: {count}")
        shown_candidates = candidates if verbose_candidates else candidates[:3]
        for match in shown_candidates:
            mapping = ", ".join(
                f"{source}->{target}"
                for source, target in sorted(match.register_mapping.items())
            )
            print(f"  {match.operation} {match.address} <{match.symbol}> [{mapping}]")
            if match.external_inputs:
                inputs = ", ".join(match.external_inputs)
                print(f"    requires external value provenance: {inputs}")
        if not verbose_candidates and len(candidates) > len(shown_candidates):
            print(
                f"  ... {len(candidates) - len(shown_candidates)} more; "
                "use --verbose-candidates or inspect the JSON report"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("aarch64", "x86_64"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--require-family",
        action="append",
        choices=("fp128", "fp64"),
        default=[],
    )
    parser.add_argument(
        "--report-family",
        action="append",
        choices=("fp128", "fp64"),
        default=[],
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--llvm-objdump")
    parser.add_argument("--verbose-candidates", action="store_true")
    parser.add_argument(
        "--proof-object",
        action="append",
        type=parse_proof_object_argument,
        default=[],
        metavar="FAMILY:OPERATION=PATH",
        help=(
            "report decoded instruction-template matches under an injective "
            "register renaming; these are candidates, not completed proofs"
        ),
    )
    args = parser.parse_args()

    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")
    families = list(dict.fromkeys([*args.require_family, *args.report_family]))
    if not families:
        parser.error("select at least one --require-family or --report-family")

    tool = fp128.find_llvm_objdump(args.llvm_objdump)
    patterns = expected_patterns(args.architecture)
    selected = {family: patterns[family] for family in families}
    matches = find_matches_in_symbols(
        iter_disassembly_symbols(tool, args.binary, args.architecture), selected
    )
    templates = load_structural_templates(
        tool, args.architecture, args.proof_object
    )
    structural_matches = find_structural_matches_in_symbols(
        iter_disassembly_symbols(tool, args.binary, args.architecture),
        templates,
        args.architecture,
    )
    result = report(
        args.binary,
        args.architecture,
        matches,
        structural_matches,
        families,
        args.verbose_candidates,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    for family in args.require_family:
        validate_required(family, matches)


if __name__ == "__main__":
    main()
