//! Generates Lean expansion programs from the Jolt bytecode expander.
//!
//! The generator runs the real Rust expansion (`expand_instruction`) and renders
//! the resulting final Jolt rows as Lean `Program` definitions.
//!
//! Usage:
//!
//! ```text
//! cargo run -p jolt-lean-gen -- LB
//! cargo run -p jolt-lean-gen -- --all
//! cargo run -p jolt-lean-gen -- --lean --out /path/to/ExpansionsAutomated.lean
//! ```

#![expect(clippy::print_stdout, reason = "CLI tool: stdout is the output")]

use std::{fmt::Write as _, fs, path::Path};

use jolt_common::constants::RISCV_REGISTER_COUNT;
use jolt_program::expand::{
    expand_instruction, is_source_only, ExpansionAllocator, ExpansionError,
};
use jolt_riscv::{
    JoltInstructionRow, NormalizedOperands, SourceInstruction, SourceInstructionKind,
    SourceInstructionRow, RV64IMAC_JOLT,
};

const DEFAULT_LEAN_OUTPUT: &str =
    "/Users/ari.biswas/Lean/lz-qed/JoltBytecode/JoltISA/ExpansionsAutomated.lean";
const SOURCE_ADDRESS: usize = 0x8000_0000;

const SOURCE_RD: u8 = 1;
const SOURCE_RS1: u8 = 2;
const SOURCE_RS2: u8 = 3;

// Because word shifts (SLLIW, SRLIW, SRAIW) have a 5-bit shift amount — valid range 0–31.
// Using 37 as sample_imm would be out of range and might produce a different expansion structure.
// So smaller values (5 and 13) are used that fit within the valid 5-bit range.
const SHIFT64_SAMPLE_IMM: i128 = 37;
const SHIFT64_CHECK_IMM: i128 = 41;
const SHIFT32_SAMPLE_IMM: i128 = 5;
const SHIFT32_CHECK_IMM: i128 = 13;

#[derive(Clone, Copy)]
struct SourceShape {
    rd: bool,
    rs1: bool,
    rs2: bool,
    imm: bool,
}

struct ExpansionArm {
    rows: Vec<JoltInstructionRow>,
    source_imm_flags: Vec<bool>,
}

fn usage() -> &'static str {
    "usage: jolt-lean-gen <INSTRUCTION|--all|--lean [--out PATH]>  (e.g. LB)"
}

fn reg_symbol(reg: u8) -> String {
    match reg {
        0 => "(regidx.Regidx 0)".to_string(),
        SOURCE_RD => "rd".to_string(),
        SOURCE_RS1 => "rs1".to_string(),
        SOURCE_RS2 => "rs2".to_string(),
        v if v >= RISCV_REGISTER_COUNT => format!("(BitVec.ofNat 7 {v})"),
        x => format!("(regidx.Regidx {x})"),
    }
}

fn operand(reg: Option<u8>) -> String {
    match reg {
        None => "?".to_string(),
        Some(v) if v >= RISCV_REGISTER_COUNT => format!("(.vreg {})", reg_symbol(v)),
        Some(x) => format!("(.xreg {})", reg_symbol(x)),
    }
}

fn regidx(reg: Option<u8>) -> String {
    reg.map_or_else(|| "?".to_string(), reg_symbol)
}

fn imm_is_nat(name: &str) -> bool {
    matches!(
        name,
        "VirtualSRLI"
            | "VirtualSRAI"
            | "VirtualROTRI"
            | "VirtualROTRIW"
            | "VirtualPow2I"
            | "VirtualPow2IW"
            | "VirtualShiftRightBitmaskI"
    )
}

fn imm_width(name: &str) -> u32 {
    match name {
        "BEQ" | "BNE" | "BLT" | "BGE" | "BLTU" | "BGEU" | "VirtualAssertEQ" => 13,
        "AUIPC" => 20,
        "JAL" => 21,
        "LUI" | "VirtualMULI" | "VirtualAdvice" | "VirtualAdviceLoad" | "VirtualAdviceLen" => 64,
        _ => 12,
    }
}

fn lean_imm(name: &str, imm: i128, override_value: Option<&str>) -> String {
    if let Some(value) = override_value {
        return value.to_string();
    }

    let bits = imm as u64;
    match name {
        "VirtualSRAI" => return format!("(sraiBitmask {})", bits.trailing_zeros()),
        "VirtualSRLI" => return format!("(srliBitmask {})", bits.trailing_zeros()),
        "VirtualMULI" if bits.is_power_of_two() => {
            return format!("(slliMultiplier {})", bits.trailing_zeros());
        }
        _ => {}
    }

    if imm_is_nat(name) {
        format!("{bits}")
    } else {
        format!("({} : BitVec {})", imm as i64, imm_width(name))
    }
}

fn lean_instr(
    row: &JoltInstructionRow,
    imm_override: Option<&str>,
    load_class: &str,
    align_fault: &str,
) -> String {
    let name = row.instruction_kind.name();
    let o = &row.operands;
    let rd = operand(o.rd);
    let rs1 = operand(o.rs1);
    let rs2 = operand(o.rs2);
    let imm = lean_imm(name, o.imm, imm_override);

    let args = match name {
        "ADD"
        | "SUB"
        | "MUL"
        | "MULHU"
        | "MULHSU"
        | "OR"
        | "XOR"
        | "AND"
        | "SLT"
        | "SLTU"
        | "ANDN"
        | "VirtualChangeDivisor"
        | "VirtualChangeDivisorW"
        | "VirtualSRL"
        | "VirtualSRA" => format!("{rd} {rs1} {rs2}"),
        "ADDI" | "ANDI" | "ORI" | "XORI" | "SLTI" | "SLTIU" | "VirtualMULI" | "VirtualSRLI"
        | "VirtualSRAI" | "VirtualROTRI" | "VirtualROTRIW" => {
            format!("{rd} {rs1} {imm}")
        }
        "VirtualPow2"
        | "VirtualPow2W"
        | "VirtualShiftRightBitmask"
        | "VirtualSignExtendWord"
        | "VirtualZeroExtendWord"
        | "VirtualMovsign"
        | "VirtualRev8W" => format!("{rd} {rs1}"),
        "LD" => format!(".{load_class} {rd} {rs1} {imm}"),
        "VirtualAssertWordAlignment" | "VirtualAssertHalfwordAlignment" => {
            format!("{} {imm} {align_fault}", regidx(o.rs1))
        }
        "SD" | "SB" | "SH" | "SW" => format!("{rs1} {rs2} {imm}"),
        "VirtualAdvice" | "VirtualAdviceLoad" | "VirtualAdviceLen" => format!("{rd} {imm}"),
        "AUIPC" | "JAL" | "LUI" => format!("{rd} {imm}"),
        "JALR" => format!("{rd} {rs1} {imm}"),
        "VirtualAssertValidDiv0"
        | "VirtualAssertValidUnsignedRemainder"
        | "VirtualAssertMulUNoOverflow"
        | "VirtualAssertLTE" => format!("{rs1} {rs2}"),
        "VirtualAssertEQ" => format!("{rs1} {rs2} {imm}"),
        "NoOp" | "FENCE" | "VirtualHostIO" => String::new(),
        _ => {
            let mut parts = Vec::new();
            if o.rd.is_some() {
                parts.push(rd);
            }
            if o.rs1.is_some() {
                parts.push(rs1);
            }
            if o.rs2.is_some() {
                parts.push(rs2);
            }
            parts.push(imm);
            parts.join(" ")
        }
    };

    if args.is_empty() {
        format!(".instr (.{name}) <|")
    } else {
        format!(".instr (.{name} {args}) <|")
    }
}

fn kind_from_name(name: &str) -> Option<SourceInstructionKind> {
    SourceInstructionKind::ALL
        .iter()
        .copied()
        .find(|kind| kind.name() == name)
}

enum Class {
    Expand,
    Unsupported,
    NotExpandable,
}

fn is_unsupported(kind: SourceInstructionKind) -> bool {
    matches!(
        kind.name(),
        "ECALL" | "EBREAK" | "MRET" | "CSRRW" | "CSRRS" | "LRW" | "LRD" | "SCW" | "SCD" | "Inline"
    )
}

fn classify(kind: SourceInstructionKind) -> Class {
    if !is_source_only(kind) {
        Class::NotExpandable
    } else if is_unsupported(kind) {
        Class::Unsupported
    } else {
        Class::Expand
    }
}

fn is_load(name: &str) -> bool {
    matches!(name, "LB" | "LBU" | "LH" | "LHU" | "LW" | "LWU")
}

fn is_store(name: &str) -> bool {
    matches!(name, "SB" | "SH" | "SW")
}

fn is_advice_load(name: &str) -> bool {
    matches!(name, "AdviceLB" | "AdviceLH" | "AdviceLW" | "AdviceLD")
}

fn is_shift_imm(name: &str) -> bool {
    matches!(name, "SLLI" | "SRLI" | "SRAI" | "SLLIW" | "SRLIW" | "SRAIW")
}

fn is_word_shift_imm(name: &str) -> bool {
    matches!(name, "SLLIW" | "SRLIW" | "SRAIW")
}

fn has_symbolic_source_imm(kind: SourceInstructionKind) -> bool {
    let name = kind.name();
    name == "ADDIW" || is_load(name) || is_store(name)
}

fn source_shape(kind: SourceInstructionKind) -> SourceShape {
    let name = kind.name();
    if is_store(name) {
        SourceShape {
            rd: false,
            rs1: true,
            rs2: true,
            imm: true,
        }
    } else if is_load(name) {
        SourceShape {
            rd: true,
            rs1: true,
            rs2: false,
            imm: true,
        }
    } else if is_advice_load(name) {
        SourceShape {
            rd: true,
            rs1: false,
            rs2: false,
            imm: false,
        }
    } else if name == "ADDIW" || is_shift_imm(name) {
        SourceShape {
            rd: true,
            rs1: true,
            rs2: false,
            imm: true,
        }
    } else {
        SourceShape {
            rd: true,
            rs1: true,
            rs2: true,
            imm: false,
        }
    }
}

fn source_sample_imm(kind: SourceInstructionKind) -> i128 {
    if is_word_shift_imm(kind.name()) {
        SHIFT32_SAMPLE_IMM
    } else {
        SHIFT64_SAMPLE_IMM
    }
}

fn source_check_imm(kind: SourceInstructionKind) -> i128 {
    if is_word_shift_imm(kind.name()) {
        SHIFT32_CHECK_IMM
    } else {
        SHIFT64_CHECK_IMM
    }
}

fn source_instruction(kind: SourceInstructionKind, rd: u8, imm: i128) -> SourceInstruction {
    let shape = source_shape(kind);
    SourceInstruction::new(
        kind,
        SourceInstructionRow {
            address: SOURCE_ADDRESS,
            operands: NormalizedOperands {
                rd: shape.rd.then_some(rd),
                rs1: shape.rs1.then_some(SOURCE_RS1),
                rs2: shape.rs2.then_some(SOURCE_RS2),
                imm: if shape.imm { imm } else { 0 },
            },
            inline: None,
            is_compressed: false,
        },
    )
}

fn expand_rows(
    kind: SourceInstructionKind,
    rd: u8,
    imm: i128,
) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let input = source_instruction(kind, rd, imm);
    expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)
        .map(|expanded| expanded.into_iter().map(JoltInstructionRow::from).collect())
}

fn same_row_except_imm(lhs: &JoltInstructionRow, rhs: &JoltInstructionRow) -> bool {
    lhs.instruction_kind == rhs.instruction_kind
        && lhs.address == rhs.address
        && lhs.operands.rd == rhs.operands.rd
        && lhs.operands.rs1 == rhs.operands.rs1
        && lhs.operands.rs2 == rhs.operands.rs2
        && lhs.virtual_sequence_remaining == rhs.virtual_sequence_remaining
        && lhs.is_first_in_sequence == rhs.is_first_in_sequence
        && lhs.is_compressed == rhs.is_compressed
}

fn source_imm_flags(
    kind: SourceInstructionKind,
    rows: &[JoltInstructionRow],
    check_rows: &[JoltInstructionRow],
    sample_imm: i128,
    check_imm: i128,
) -> Result<Vec<bool>, String> {
    if rows.len() != check_rows.len() {
        return Err(format!(
            "source-immediate check changed expansion length: {} vs {} rows",
            rows.len(),
            check_rows.len()
        ));
    }

    rows.iter()
        .zip(check_rows)
        .enumerate()
        .map(|(index, (row, check_row))| {
            if !same_row_except_imm(row, check_row) {
                return Err(format!(
                    "source-immediate check changed row {index}: {:?} vs {:?}",
                    row, check_row
                ));
            }

            // if the row.operands.imm is sample_imm
            // and check_row.operands.imm is check_imm
            // then the imm in this instruction is NOT hardcoded, and it's the user specified imm.
            // In this case the lean generator should use "imm" instead of the actual hard-coded
            // constant in this case.
            let tracks_source_imm =
                row.operands.imm == sample_imm && check_row.operands.imm == check_imm;
            if row.operands.imm == sample_imm && !tracks_source_imm {
                return Err(format!(
                    "immediate sample collision at expanded row {index} ({:?}): \
                     {sample_imm} did not track source immediate check value {check_imm}",
                    row.instruction_kind
                ));
            }

            Ok(tracks_source_imm && has_symbolic_source_imm(kind))
        })
        .collect()
}

fn expansion_arm(
    kind: SourceInstructionKind,
    rd: u8,
) -> Result<ExpansionArm, Box<dyn std::error::Error>> {
    let sample_imm = source_sample_imm(kind);
    let check_imm = source_check_imm(kind);
    let rows = expand_rows(kind, rd, sample_imm)?;
    let check_rows = expand_rows(kind, rd, check_imm)?;
    let source_imm_flags = source_imm_flags(kind, &rows, &check_rows, sample_imm, check_imm)?;
    Ok(ExpansionArm {
        rows,
        source_imm_flags,
    })
}

fn load_class_and_align_fault(kind: SourceInstructionKind) -> (&'static str, &'static str) {
    let name = kind.name();
    let write_or_atomic = name.starts_with("AMO")
        || name.starts_with("LR")
        || matches!(name, "SB" | "SH" | "SW" | "SD" | "SCD" | "SCW");
    let align_fault = if write_or_atomic {
        "(ExceptionType.E_SAMO_Addr_Align ())"
    } else {
        "(ExceptionType.E_Load_Addr_Align ())"
    };

    let atomic = name.starts_with("AMO") || name.starts_with("LR") || matches!(name, "SCD" | "SCW");
    let load_class = if atomic { "amo" } else { "normal" };
    (load_class, align_fault)
}

fn is_advice_row(name: &str) -> bool {
    matches!(
        name,
        "VirtualAdvice" | "VirtualAdviceLoad" | "VirtualAdviceLen"
    )
}

fn render_rows(
    arm: &ExpansionArm,
    load_class: &str,
    align_fault: &str,
    indent: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut out = String::new();
    let padding = " ".repeat(indent);
    let mut advice_index = 0usize;

    for (instruction, source_imm) in arm.rows.iter().zip(&arm.source_imm_flags) {
        let advice_override = if is_advice_row(instruction.instruction_kind.name()) {
            let value = format!("advice{advice_index}");
            advice_index += 1;
            Some(value)
        } else {
            None
        };
        let source_imm_override = source_imm.then(|| "imm".to_string());
        let imm_override = advice_override
            .as_deref()
            .or(source_imm_override.as_deref());

        writeln!(
            out,
            "{padding}{}",
            lean_instr(instruction, imm_override, load_class, align_fault)
        )?;
    }
    writeln!(out, "{padding}.done RETIRE_SUCCESS")?;
    Ok(out)
}

fn emit_program(kind: SourceInstructionKind) -> Result<(), Box<dyn std::error::Error>> {
    let name = kind.name();
    let shape = source_shape(kind);
    let (load_class, align_fault) = load_class_and_align_fault(kind);

    println!("--- {name} ---");
    if shape.rd {
        for (label, rd) in [("rd == x0", 0u8), ("rd != x0", SOURCE_RD)] {
            let arm = expansion_arm(kind, rd)?;
            println!("{label}:");
            print!("{}", render_rows(&arm, load_class, align_fault, 2)?);
        }
    } else {
        let arm = expansion_arm(kind, SOURCE_RD)?;
        println!("no rd:");
        print!("{}", render_rows(&arm, load_class, align_fault, 2)?);
    }
    Ok(())
}

fn emit_all() -> Result<(), Box<dyn std::error::Error>> {
    let mut ok = 0usize;
    let mut failed = Vec::new();
    for &kind in SourceInstructionKind::ALL {
        if !matches!(classify(kind), Class::Expand) {
            continue;
        }
        match emit_program(kind) {
            Ok(()) => {
                ok += 1;
                println!();
            }
            Err(error) => failed.push(format!("{}: {error}", kind.name())),
        }
    }
    println!("-- generated {ok} expandable instruction(s)");
    if !failed.is_empty() {
        println!("-- failed ({}):", failed.len());
        for failure in &failed {
            println!("--   {failure}");
        }
    }
    Ok(())
}

fn lean_def_name(kind: SourceInstructionKind) -> String {
    format!("{}ProgramAuto", kind.name().to_ascii_lowercase())
}

fn advice_count(arm: &ExpansionArm) -> usize {
    arm.rows
        .iter()
        .filter(|row| is_advice_row(row.instruction_kind.name()))
        .count()
}

fn lean_signature(kind: SourceInstructionKind, advice_count: usize) -> String {
    let shape = source_shape(kind);
    let mut groups = Vec::new();
    let mut regs = Vec::new();

    if shape.rd {
        regs.push("rd");
    }
    if shape.rs1 {
        regs.push("rs1");
    }
    if shape.rs2 {
        regs.push("rs2");
    }
    if !regs.is_empty() {
        groups.push(format!("({} : regidx)", regs.join(" ")));
    }
    if has_symbolic_source_imm(kind) {
        groups.push("(imm : BitVec 12)".to_string());
    }
    if advice_count > 0 {
        let advice_names = (0..advice_count)
            .map(|index| format!("advice{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        groups.push(format!("({advice_names} : BitVec 64)"));
    }

    groups.join(" ")
}

fn render_definition(kind: SourceInstructionKind) -> Result<String, Box<dyn std::error::Error>> {
    let shape = source_shape(kind);
    let (load_class, align_fault) = load_class_and_align_fault(kind);
    let mut out = String::new();

    let rd_zero = shape.rd.then(|| expansion_arm(kind, 0u8)).transpose()?;
    let normal = expansion_arm(kind, SOURCE_RD)?;
    let advice_params = rd_zero.as_ref().map_or_else(
        || advice_count(&normal),
        |arm| advice_count(arm).max(advice_count(&normal)),
    );

    writeln!(
        out,
        "/-- Auto-generated from the Rust `{}` expansion. -/",
        kind.name()
    )?;
    writeln!(
        out,
        "def {} {} : Program :=",
        lean_def_name(kind),
        lean_signature(kind, advice_params)
    )?;

    if let Some(rd_zero) = rd_zero {
        writeln!(out, "  if isX0 rd then")?;
        out.push_str(&render_rows(&rd_zero, load_class, align_fault, 4)?);
        writeln!(out, "  else")?;
        out.push_str(&render_rows(&normal, load_class, align_fault, 4)?);
    } else {
        out.push_str(&render_rows(&normal, load_class, align_fault, 2)?);
    }
    writeln!(out)?;

    Ok(out)
}

fn render_lean_file() -> Result<String, Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str(
        "\
import JoltBytecode.JoltISA.Instruction
import JoltBytecode.JoltISA.VirtualRegisters
import JoltBytecode.JoltISA.Expansions.ALU

/-!
# Auto-generated Jolt-ISA expansion programs

Generated by `jolt-lean-gen --lean` from the Rust bytecode expander
(`expand_instruction`), the source of truth. Do not edit by hand;
regenerate instead. Each `...ProgramAuto` mirrors the final Jolt rows
emitted by Rust for the representative source instruction.
-/

open Sail PreSail LeanRV64D.Functions

namespace JoltISA

",
    );

    let mut generated = 0usize;
    for &kind in SourceInstructionKind::ALL {
        if !matches!(classify(kind), Class::Expand) {
            continue;
        }
        out.push_str(&render_definition(kind)?);
        generated += 1;
    }

    out.push_str("end JoltISA\n\n");
    writeln!(out, "-- generated {generated} expansion program(s)")?;
    Ok(out)
}

fn write_lean_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let lean = render_lean_file()?;
    fs::write(path, lean)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn parse_lean_output_path<I>(args: I) -> Result<String, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut output_path = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" | "-o" => {
                let path = args.next().ok_or("--lean --out requires a path argument")?;
                if output_path.replace(path).is_some() {
                    return Err("--lean output path was provided more than once".into());
                }
            }
            value if output_path.is_none() => {
                output_path = Some(value.to_string());
            }
            _ => {
                return Err(format!("unexpected --lean argument: {arg}").into());
            }
        }
    }

    Ok(output_path.unwrap_or_else(|| DEFAULT_LEAN_OUTPUT.to_string()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let arg = args.next().ok_or_else(usage)?;
    match arg.as_str() {
        "--all" => {
            if let Some(extra) = args.next() {
                return Err(format!("unexpected argument after --all: {extra}").into());
            }
            emit_all()
        }
        "--lean" => {
            let output_path = parse_lean_output_path(args)?;
            write_lean_file(Path::new(&output_path))
        }
        "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        instruction_name => {
            if let Some(extra) = args.next() {
                return Err(
                    format!("unexpected argument after {instruction_name}: {extra}").into(),
                );
            }
            let kind = kind_from_name(instruction_name)
                .ok_or_else(|| format!("unknown instruction: {instruction_name}"))?;
            match classify(kind) {
                Class::Expand => emit_program(kind),
                Class::Unsupported => {
                    println!("-- unsupported (hand-coded for now): {}", kind.name());
                    Ok(())
                }
                Class::NotExpandable => Err(format!(
                    "{}: not a source-only expansion (native or non-ISA)",
                    kind.name()
                )
                .into()),
            }
        }
    }
}
