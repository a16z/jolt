//! Generates Lean expansion programs from the Jolt bytecode expander.
//!
//! Runs the real expansion (`expand_instruction`) for one instruction and prints
//! it as a Lean `Program`: a chain of `.instr (.<Opcode> …) <|` lines ending in
//! `.done RETIRE_SUCCESS`, mirroring `JoltISA/Instruction.lean`. Both arms of the
//! `rd==x0` branch are shown.
//!
//! Usage: `cargo run -p jolt-lean-gen -- LB`  (prints to stdout for now).

// This is a CLI generator whose job is to print the result to stdout.
#![expect(clippy::print_stdout, reason = "CLI tool: stdout is the output")]

use common::constants::RISCV_REGISTER_COUNT;
use jolt_program::expand::{
    expand_instruction, is_source_only, ExpansionAllocator, ExpansionError,
};
use jolt_riscv::{
    JoltInstructionRow, NormalizedOperands, SourceInstruction, SourceInstructionKind,
    SourceInstructionRow, RV64IMAC_JOLT,
};

// Symbol for a register number: sentinels 1/2/3 are the source operands
// rd/rs1/rs2, 0 is x0, anything past the architectural registers is a virtual
// register `vN`, else a plain `xN`.
fn reg_symbol(reg: u8) -> String {
    match reg {
        0 => "x0".to_string(),
        1 => "rd".to_string(),
        2 => "rs1".to_string(),
        3 => "rs2".to_string(),
        v if v >= RISCV_REGISTER_COUNT => format!("v{v}"),
        x => format!("x{x}"),
    }
}

// Render a register as a Lean `Src`/`Dst` operand (`.vreg`/`.xreg`).
fn operand(reg: Option<u8>) -> String {
    match reg {
        None => "?".to_string(),
        Some(v) if v >= RISCV_REGISTER_COUNT => format!("(.vreg {})", reg_symbol(v)),
        Some(x) => format!("(.xreg {})", reg_symbol(x)),
    }
}

// Render a register as a bare Lean `regidx` (no `.xreg` wrapper), used where an
// `Instr` field is a plain `regidx` (e.g. alignment-assert base).
fn regidx(reg: Option<u8>) -> String {
    reg.map_or_else(|| "?".to_string(), reg_symbol)
}

// Distinctive source-immediate sentinel: valid as a 6-bit shift amount and a
// 12-bit immediate, and unlikely to collide with a recipe constant. Any row whose
// immediate equals this is the source immediate passed through, printed as `imm`.
const SRC_IMM: i128 = 37;

// Opcodes whose Lean immediate is a `Nat` (unsigned bitmask/shift), not a `BitVec`.
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

// Bit-width of an opcode's `BitVec` immediate in `Instruction.lean`.
fn imm_width(name: &str) -> u32 {
    match name {
        "BEQ" | "BNE" | "BLT" | "BGE" | "BLTU" | "BGEU" | "VirtualAssertEQ" => 13,
        "AUIPC" => 20,
        "JAL" => 21,
        "LUI" | "VirtualMULI" | "VirtualAdvice" | "VirtualAdviceLoad" | "VirtualAdviceLen" => 64,
        _ => 12,
    }
}

// Render an immediate in Lean notation, recovering the symbolic form where the
// hand-written Lean uses a helper. Same value as the concrete number, just the
// representation the Lean expects (verified equal bit-for-bit).
fn lean_imm(name: &str, imm: i128) -> String {
    if imm == SRC_IMM {
        return "imm".to_string();
    }
    let bits = imm as u64;
    match name {
        // ((1 << (64-shamt)) - 1) << shamt: trailing-zero count recovers shamt.
        "VirtualSRAI" => return format!("(sraiBitmask {})", bits.trailing_zeros()),
        "VirtualSRLI" => return format!("(srliBitmask {})", bits.trailing_zeros()),
        // 2^shamt multiplier.
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

// Build the Lean `.instr (.<Opcode> …) <|` line for one final row. The operand
// order per opcode mirrors the `Instr` constructors in `Instruction.lean`.
// `load_class` ("normal"/"amo") is the `LD` Sail fault class; `align_fault` is the
// `ExceptionType` for alignment asserts. Both are derived from the source kind.
fn lean_instr(row: &JoltInstructionRow, load_class: &str, align_fault: &str) -> String {
    let name = row.instruction_kind.name();
    let o = &row.operands;
    let rd = operand(o.rd);
    let rs1 = operand(o.rs1);
    let rs2 = operand(o.rs2);
    let imm = lean_imm(name, o.imm);

    let args: String = match name {
        // dst, lhs, rhs
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
        // dst, src, imm
        "ADDI" | "ANDI" | "ORI" | "XORI" | "SLTI" | "SLTIU" | "VirtualMULI" | "VirtualSRLI"
        | "VirtualSRAI" | "VirtualROTRI" | "VirtualROTRIW" => format!("{rd} {rs1} {imm}"),
        // dst, src
        "VirtualPow2"
        | "VirtualPow2W"
        | "VirtualShiftRightBitmask"
        | "VirtualSignExtendWord"
        | "VirtualZeroExtendWord"
        | "VirtualMovsign"
        | "VirtualRev8W" => format!("{rd} {rs1}"),
        // faultClass, dst, base, imm
        "LD" => format!(".{load_class} {rd} {rs1} {imm}"),
        // base (bare regidx), imm, fault
        "VirtualAssertWordAlignment" | "VirtualAssertHalfwordAlignment" => {
            format!("{} {imm} {align_fault}", regidx(o.rs1))
        }
        // base, value, imm (no dst)
        "SD" | "SB" | "SH" | "SW" => format!("{rs1} {rs2} {imm}"),
        // dst, value
        "VirtualAdvice" | "VirtualAdviceLoad" | "VirtualAdviceLen" => format!("{rd} {imm}"),
        // dst, imm
        "AUIPC" | "JAL" | "LUI" => format!("{rd} {imm}"),
        // dst, base, imm
        "JALR" => format!("{rd} {rs1} {imm}"),
        // two srcs, no dst
        "VirtualAssertValidDiv0"
        | "VirtualAssertValidUnsignedRemainder"
        | "VirtualAssertMulUNoOverflow"
        | "VirtualAssertLTE" => format!("{rs1} {rs2}"),
        // two srcs, imm
        "VirtualAssertEQ" => format!("{rs1} {rs2} {imm}"),
        // nullary
        "NoOp" | "FENCE" | "VirtualHostIO" => String::new(),
        // Best-effort for opcodes not yet in the table: present regs then imm.
        // Grow the arms above as new instructions are covered.
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

// Find the source instruction kind whose mnemonic matches `name` (e.g. "LB").
fn kind_from_name(name: &str) -> Option<SourceInstructionKind> {
    SourceInstructionKind::ALL
        .iter()
        .copied()
        .find(|kind| kind.name() == name)
}

// How an instruction is handled by the generator.
enum Class {
    // Source-only expansion we translate to Lean.
    Expand,
    // Source-only, but hand-coded in Lean for now — we don't generate these yet.
    Unsupported,
    // Not source-only: the expander takes the native path (a native Jolt op) or the
    // kind is synthetic / non-ISA. Nothing for us to expand.
    NotExpandable,
}

// Source-only kinds we deliberately leave hand-written for now.
fn is_unsupported(kind: SourceInstructionKind) -> bool {
    matches!(
        kind.name(),
        // System / CSR
        "ECALL" | "EBREAK" | "MRET" | "CSRRW" | "CSRRS"
        // Load-reserved / store-conditional (hand-coded in LoadReserved.lean)
        | "LRW" | "LRD" | "SCW" | "SCD"
        // Registered inline dispatch (needs an InlineExpansionProvider)
        | "Inline"
    )
}

fn classify(kind: SourceInstructionKind) -> Class {
    // Mirror the expander's own decision (`dispatch_source`): only `is_source_only`
    // kinds get expanded; everything else takes the native path.
    if !is_source_only(kind) {
        Class::NotExpandable
    } else if is_unsupported(kind) {
        Class::Unsupported
    } else {
        Class::Expand
    }
}

// Expand `kind` and print it as a Lean `Program`, both arms of the rd==x0 branch.
fn emit_program(kind: SourceInstructionKind) -> Result<(), ExpansionError> {
    let n = kind.name();
    // The top-level alignment assert reports a store/AMO align fault for any
    // memory write or atomic, and a load align fault for a plain load.
    let write_or_atomic = n.starts_with("AMO")
        || n.starts_with("LR")
        || matches!(n, "SB" | "SH" | "SW" | "SD" | "SCD" | "SCW");
    let align_fault = if write_or_atomic {
        "(ExceptionType.E_SAMO_Addr_Align ())"
    } else {
        "(ExceptionType.E_Load_Addr_Align ())"
    };
    // The internal aligned-doubleword `LD` is the read side of an atomic (`.amo`)
    // only for true atomics. A plain store's read-modify-write read is a normal
    // load: its address is always 8-aligned, so the fault class is unreachable at
    // runtime (`Semantics.lean` consults it only on a misaligned address) and
    // merely selects the Sail exception class the proof layer expects there.
    // Match the hand-written Lean, which uses `.normal` for plain stores.
    let atomic = n.starts_with("AMO") || n.starts_with("LR") || matches!(n, "SCD" | "SCW");
    let load_class = if atomic { "amo" } else { "normal" };

    println!("--- {n} ---");
    // Feed sentinel operands (rd=1, rs1=2, rs2=3, imm=SRC_IMM), once with rd==x0
    // and once without, so the rd==x0 branch is shown.
    for (arm, rd) in [("rd == x0", 0u8), ("rd != x0", 1u8)] {
        let mut allocator = ExpansionAllocator::new();
        let row = SourceInstructionRow {
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: Some(2),
                rs2: Some(3),
                imm: SRC_IMM,
            },
            inline: None,
            is_compressed: false,
        };
        let input = SourceInstruction::new(kind, row);
        let expanded = expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)?;
        println!("{arm}:");
        for instruction in expanded {
            println!(
                "  {}",
                lean_instr(
                    &JoltInstructionRow::from(instruction),
                    load_class,
                    align_fault
                )
            );
        }
        println!("  .done RETIRE_SUCCESS");
    }
    Ok(())
}

// Emit every expandable instruction, then a summary of coverage and failures.
// A thoroughness pass: surfaces any instruction that errors under expansion.
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
            Err(e) => failed.push(format!("{}: {e}", kind.name())),
        }
    }
    println!("-- generated {ok} expandable instruction(s)");
    if !failed.is_empty() {
        println!("-- failed ({}):", failed.len());
        for f in &failed {
            println!("--   {f}");
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args()
        .nth(1)
        .ok_or("usage: gen-lean <INSTRUCTION|--all>  (e.g. LB)")?;
    if arg == "--all" {
        return emit_all();
    }

    let kind = kind_from_name(&arg).ok_or_else(|| format!("unknown instruction: {arg}"))?;
    match classify(kind) {
        Class::Expand => emit_program(kind)?,
        Class::Unsupported => {
            println!("-- unsupported (hand-coded for now): {}", kind.name());
        }
        Class::NotExpandable => {
            return Err(format!(
                "{}: not a source-only expansion (native or non-ISA)",
                kind.name()
            )
            .into())
        }
    }
    Ok(())
}
