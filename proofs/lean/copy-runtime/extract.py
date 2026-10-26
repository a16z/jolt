#!/usr/bin/env python3
"""Generate the Lean data files for the copy runtime proof.

usage: extract.py OBJECT LISTING SOURCE OUT_DIR

OBJECT   ELF64 RISC-V relocatable object assembled from jolt-platform/src/memory.S
LISTING  output of `llvm-objdump -d -r --mattr=+c -M no-aliases -M numeric OBJECT`
SOURCE   the assembly source (only hashed, to record what was assembled)
OUT_DIR  directory receiving Object.lean and Disasm.lean

Object.lean is produced from the ELF file alone (section bytes, RELA entries,
symbols). Disasm.lean is produced from the llvm-objdump listing alone: every
disassembled instruction is translated from LLVM's textual form into the Lean
instruction syntax, applying the RVC expansions of the ISA manual. The Lean
kernel then checks that the hand-written Lean decoder, applied to the bytes in
Object.lean, reproduces Disasm.lean exactly. Both outputs are data only.
"""

import hashlib
import re
import struct
import sys
from pathlib import Path

# ELF section name -> Lean identifier. The extraction fails if the object has
# any other non-empty executable section, so nothing executable is overlooked.
SECTIONS = {
    ".text.__wrap_memcpy": "wrapMemcpy",
    ".text.__wrap_memmove": "wrapMemmove",
    ".text.__jolt_copy_misaligned": "joltCopyMisaligned",
}

R_RISCV_JAL = 17
R_RISCV_CALL_PLT = 19
RELOC_KINDS = {R_RISCV_JAL: ".jal", R_RISCV_CALL_PLT: ".callPlt"}
RELOC_NAMES = {"R_RISCV_JAL": R_RISCV_JAL, "R_RISCV_CALL_PLT": R_RISCV_CALL_PLT}

SHT_SYMTAB = 2
SHT_RELA = 4
SHT_REL = 9
SHF_EXECINSTR = 0x4
EM_RISCV = 243
EF_RISCV_RVC = 0x1


def fail(message):
    sys.exit(f"extract.py: {message}")


def c_string(table, offset):
    end = table.index(b"\0", offset)
    return table[offset:end].decode("ascii")


def parse_elf(data):
    if data[:4] != b"\x7fELF":
        fail("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        fail("expected ELFCLASS64 little-endian")
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    if e_type != 1 or e_machine != EM_RISCV:
        fail("expected a RISC-V relocatable object")
    (e_flags,) = struct.unpack_from("<I", data, 48)
    if not e_flags & EF_RISCV_RVC:
        fail("object is not marked EF_RISCV_RVC")
    (e_shoff,) = struct.unpack_from("<Q", data, 40)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 58)
    if e_shentsize != 64:
        fail("unexpected section header size")

    headers = []
    for index in range(e_shnum):
        fields = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + index * 64)
        headers.append(
            dict(
                zip(
                    (
                        "name",
                        "type",
                        "flags",
                        "addr",
                        "offset",
                        "size",
                        "link",
                        "info",
                        "addralign",
                        "entsize",
                    ),
                    fields,
                )
            )
        )
    shstr = headers[e_shstrndx]
    shstrtab = data[shstr["offset"] : shstr["offset"] + shstr["size"]]
    for header in headers:
        header["sname"] = c_string(shstrtab, header["name"])
        header["data"] = data[header["offset"] : header["offset"] + header["size"]]

    symtabs = [h for h in headers if h["type"] == SHT_SYMTAB]
    if len(symtabs) != 1:
        fail("expected exactly one symbol table")
    symtab = symtabs[0]
    strtab = headers[symtab["link"]]["data"]
    symbols = []
    for offset in range(0, symtab["size"], 24):
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
            "<IBBHQQ", symtab["data"], offset
        )
        symbols.append(
            {
                "name": c_string(strtab, st_name),
                "type": st_info & 0xF,
                "shndx": st_shndx,
                "value": st_value,
                "size": st_size,
            }
        )
    return headers, symbols


def extract_sections(headers, symbols):
    executable = [
        h for h in headers if h["flags"] & SHF_EXECINSTR and h["size"] > 0
    ]
    names = sorted(h["sname"] for h in executable)
    if names != sorted(SECTIONS):
        fail(f"unexpected executable sections {names}")
    if any(h["type"] == SHT_REL for h in headers):
        fail("REL relocation sections are not supported")

    result = {}
    for index, header in enumerate(headers):
        if header["sname"] not in SECTIONS:
            continue
        relocs = []
        for rela in headers:
            if rela["type"] != SHT_RELA or rela["info"] != index:
                continue
            for offset in range(0, rela["size"], 24):
                r_offset, r_info, r_addend = struct.unpack_from(
                    "<QQq", rela["data"], offset
                )
                kind = r_info & 0xFFFFFFFF
                if kind not in RELOC_KINDS:
                    fail(f"unsupported relocation type {kind} in {header['sname']}")
                symbol = symbols[r_info >> 32]
                if not symbol["name"]:
                    fail("relocation against an unnamed symbol")
                relocs.append((r_offset, kind, symbol["name"], r_addend))
        defined = [
            (s["name"], s["value"], s["size"])
            for s in symbols
            if s["shndx"] == index and s["name"] and not s["name"].startswith("$")
        ]
        result[header["sname"]] = {
            "bytes": header["data"],
            "relocs": sorted(relocs),
            "symbols": sorted(defined, key=lambda d: (d[1], d[0])),
        }
    return result


LINE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+\t(\S+)(?:\t(.*))?$")
RELOC_LINE = re.compile(r"^\s+([0-9a-f]+):\s+(R_RISCV_\w+)\s+(\S+)$")
SECTION_LINE = re.compile(r"^Disassembly of section (\S+):$")
MEM_OPERAND = re.compile(r"^(-?(?:0x[0-9a-f]+|\d+))\((x\d+)\)$")

LOAD_WIDTHS = {"lbu": ".byte", "lhu": ".half", "lwu": ".word", "ld": ".double"}
STORE_WIDTHS = {"sb": ".byte", "sh": ".half", "sw": ".word", "sd": ".double"}
BRANCHES = {"beq": ".eq", "bne": ".ne", "bltu": ".ltu", "bgeu": ".geu"}
ALU = {"add": ".add", "sub": ".sub", "or": ".or", "xor": ".xor", "sll": ".sll", "srl": ".srl"}
ALU_IMM = {"addi": ".addi", "andi": ".andi", "slli": ".slli", "srli": ".srli"}


def reg(text):
    match = re.fullmatch(r"x(\d+)", text)
    if not match or int(match.group(1)) > 31:
        fail(f"bad register {text!r}")
    return f"({int(match.group(1))} : Reg)"


def number(text):
    return int(text, 0)


def target(text, address):
    # Branch and jump operands are printed as absolute section offsets,
    # optionally followed by a symbolic annotation `<sym+0x..>`.
    return number(text.split()[0]) - address


def imm(value):
    return f"(imm ({value}))"


def memory(text):
    match = MEM_OPERAND.fullmatch(text)
    if not match:
        fail(f"bad memory operand {text!r}")
    return number(match.group(1)), reg(match.group(2))


def translate(mnemonic, operands, address):
    ops = [o.strip() for o in operands.split(",")] if operands else []
    if mnemonic in LOAD_WIDTHS or mnemonic == "c.ld":
        rd = reg(ops[0])
        offset, base = memory(ops[1])
        width = LOAD_WIDTHS.get(mnemonic, ".double")
        return f".load {width} {rd} {base} {imm(offset)}"
    if mnemonic in STORE_WIDTHS:
        rs2 = reg(ops[0])
        offset, base = memory(ops[1])
        return f".store {STORE_WIDTHS[mnemonic]} {rs2} {base} {imm(offset)}"
    if mnemonic in ALU:
        return f".alu {ALU[mnemonic]} {reg(ops[0])} {reg(ops[1])} {reg(ops[2])}"
    if mnemonic in ALU_IMM:
        return f".aluImm {ALU_IMM[mnemonic]} {reg(ops[0])} {reg(ops[1])} {imm(number(ops[2]))}"
    if mnemonic in BRANCHES:
        return (
            f".branch {BRANCHES[mnemonic]} {reg(ops[0])} {reg(ops[1])} "
            f"{imm(target(ops[2], address))}"
        )
    if mnemonic == "auipc":
        # The printed operand is the raw 20-bit field; auipc adds sext32(imm20 << 12).
        value = (number(ops[1]) << 12) & 0xFFFFFFFF
        if value >= 1 << 31:
            value -= 1 << 32
        return f".auipc {reg(ops[0])} {imm(value)}"
    if mnemonic == "jal":
        return f".jal {reg(ops[0])} {imm(target(ops[1], address))}"
    if mnemonic == "jalr":
        offset, base = memory(ops[1].split()[0])
        return f".jalr {reg(ops[0])} {base} {imm(offset)}"
    # RVC expansions (ISA manual, "C" Standard Extension).
    if mnemonic == "c.addi":
        return f".aluImm .addi {reg(ops[0])} {reg(ops[0])} {imm(number(ops[1]))}"
    if mnemonic == "c.li":
        return f".aluImm .addi {reg(ops[0])} {reg('x0')} {imm(number(ops[1]))}"
    if mnemonic == "c.andi":
        return f".aluImm .andi {reg(ops[0])} {reg(ops[0])} {imm(number(ops[1]))}"
    if mnemonic in ("c.slli", "c.srli"):
        op = ".slli" if mnemonic == "c.slli" else ".srli"
        return f".aluImm {op} {reg(ops[0])} {reg(ops[0])} {imm(number(ops[1]))}"
    if mnemonic == "c.mv":
        return f".alu .add {reg(ops[0])} {reg('x0')} {reg(ops[1])}"
    if mnemonic == "c.add":
        return f".alu .add {reg(ops[0])} {reg(ops[0])} {reg(ops[1])}"
    if mnemonic == "c.sub":
        return f".alu .sub {reg(ops[0])} {reg(ops[0])} {reg(ops[1])}"
    if mnemonic == "c.jr":
        return f".jalr {reg('x0')} {reg(ops[0])} {imm(0)}"
    if mnemonic == "c.j":
        return f".jal {reg('x0')} {imm(target(ops[0], address))}"
    if mnemonic in ("c.beqz", "c.bnez"):
        cond = ".eq" if mnemonic == "c.beqz" else ".ne"
        return f".branch {cond} {reg(ops[0])} {reg('x0')} {imm(target(ops[1], address))}"
    fail(f"no translation for {mnemonic!r} at {address:#x}")


def parse_listing(text, sections):
    tables = {name: [] for name in SECTIONS}
    listed_relocs = {name: [] for name in SECTIONS}
    current = None
    for line in text.splitlines():
        header = SECTION_LINE.match(line)
        if header:
            current = header.group(1)
            if current not in SECTIONS:
                fail(f"listing has unexpected section {current}")
            continue
        reloc = RELOC_LINE.match(line)
        if reloc and current:
            listed_relocs[current].append(
                (int(reloc.group(1), 16), RELOC_NAMES.get(reloc.group(2)), reloc.group(3))
            )
            continue
        match = LINE.match(line)
        if not match or current is None:
            continue
        address = int(match.group(1), 16)
        encoding = match.group(2)
        length = len(encoding) // 2
        if length not in (2, 4):
            fail(f"unexpected encoding {encoding} at {address:#x}")
        expected = sections[current]["bytes"][address : address + length]
        if expected[::-1].hex() != encoding:
            fail(f"listing bytes {encoding} disagree with ELF at {current}+{address:#x}")
        instr = translate(match.group(3), match.group(4) or "", address)
        tables[current].append((address, instr, length))
    for name, section in sections.items():
        elf = [(offset, kind, symbol) for offset, kind, symbol, _ in section["relocs"]]
        if sorted(listed_relocs[name]) != elf:
            fail(f"listing relocations disagree with ELF in {name}")
        covered = sum(length for _, _, length in tables[name])
        if covered != len(section["bytes"]):
            fail(f"listing covers {covered} of {len(section['bytes'])} bytes of {name}")
    return tables


HEADER = """\
-- Generated by proofs/lean/copy-runtime/extract.py from {source}
-- (sha256 {digest}). Do not edit; run ./regenerate.sh.
"""


def lean_string(text):
    if not re.fullmatch(r"[A-Za-z0-9_.$]+", text):
        fail(f"unexpected symbol name {text!r}")
    return f'"{text}"'


def write_object(path, sections, header):
    out = [header, "import CopyRuntime.Machine.Image", "", "namespace CopyRuntime.Generated", ""]
    for elf_name, lean_name in SECTIONS.items():
        section = sections[elf_name]
        data = section["bytes"]
        out.append(f"/-- `{elf_name}` ({len(data)} bytes). -/")
        out.append(f"def {lean_name} : Section where")
        out.append(f"  name := {lean_string(elf_name)}")
        out.append("  bytes := [")
        for row in range(0, len(data), 16):
            chunk = ", ".join(f"0x{b:02x}" for b in data[row : row + 16])
            comma = "," if row + 16 < len(data) else ""
            out.append(f"    {chunk}{comma}")
        out.append("  ]")
        relocs = [
            f"{{ offset := 0x{offset:x}, kind := {RELOC_KINDS[kind]}, "
            f"symbol := {lean_string(symbol)}, addend := {addend} }}"
            for offset, kind, symbol, addend in section["relocs"]
        ]
        out.append("  relocs := [" + ", ".join(relocs) + "]")
        symbols = [
            f"{{ name := {lean_string(name)}, offset := 0x{value:x}, size := {size} }}"
            for name, value, size in section["symbols"]
        ]
        out.append("  symbols := [" + ", ".join(symbols) + "]")
        out.append("")
    out.append("end CopyRuntime.Generated")
    path.write_text("\n".join(out) + "\n")


def write_disasm(path, tables, header):
    out = [
        header,
        "import CopyRuntime.Machine.Instr",
        "",
        "/-! Instruction listings translated from `llvm-objdump -M no-aliases -M numeric`.",
        "Each entry is `(section offset, instruction, length in bytes)`. -/",
        "",
        "namespace CopyRuntime.Generated.Disasm",
        "",
    ]
    for elf_name, lean_name in SECTIONS.items():
        out.append(f"def {lean_name} : List (Nat × Instr × Nat) := [")
        rows = tables[elf_name]
        for index, (address, instr, length) in enumerate(rows):
            comma = "," if index + 1 < len(rows) else ""
            out.append(f"  (0x{address:x}, {instr}, {length}){comma}")
        out.append("]")
        out.append("")
    out.append("end CopyRuntime.Generated.Disasm")
    path.write_text("\n".join(out) + "\n")


def main():
    if len(sys.argv) != 5:
        fail(__doc__.strip().splitlines()[2])
    obj, listing, source, out_dir = (Path(a) for a in sys.argv[1:])
    headers, symbols = parse_elf(obj.read_bytes())
    sections = extract_sections(headers, symbols)
    tables = parse_listing(listing.read_text(), sections)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    header = HEADER.format(source="jolt-platform/src/memory.S", digest=digest)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_object(out_dir / "Object.lean", sections, header)
    write_disasm(out_dir / "Disasm.lean", tables, header)


if __name__ == "__main__":
    main()
