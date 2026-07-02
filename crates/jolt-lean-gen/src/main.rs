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
    CircuitFlags, Flags, JoltInstruction, JoltInstructionRow, NormalizedOperands,
    SourceInstruction, SourceInstructionKind, SourceInstructionRow, RV64IMAC_JOLT,
};

// Lean term for a register number. Sentinels 1/2/3 are the source operands
// rd/rs1/rs2 (rendered as bare `def` parameters); 0 is architectural `x0`;
// virtual registers at 40+ are the allocator temps `inlineTmp 0, 1, …`
// (`inlineTmp0` = v40); reserved virtual registers (32..40) fall back to an
// explicit `BitVec`, and other architectural numbers to `regidx.Regidx`.
fn reg_symbol(reg: u8) -> String {
    const INLINE_BASE: u8 = 40;
    match reg {
        0 => "(regidx.Regidx 0)".to_string(),
        1 => "rd".to_string(),
        2 => "rs1".to_string(),
        3 => "rs2".to_string(),
        v if v >= INLINE_BASE => format!("(inlineTmp {})", v - INLINE_BASE),
        v if v >= RISCV_REGISTER_COUNT => format!("(BitVec.ofNat 7 {v})"),
        x => format!("(regidx.Regidx {x})"),
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
// `imm_override`, when set, replaces the rendered immediate — used to inject a
// symbolic advice parameter for `VirtualAdvice`/`VirtualAdviceLoad`, whose value
// is an oracle input the trace supplies (the concrete number the expander emits
// there is only a placeholder).
fn lean_instr(
    row: &JoltInstructionRow,
    load_class: &str,
    align_fault: &str,
    imm_override: Option<&str>,
) -> String {
    let name = row.instruction_kind.name();
    let o = &row.operands;
    let rd = operand(o.rd);
    let rs1 = operand(o.rs1);
    let rs2 = operand(o.rs2);
    let imm = imm_override.map_or_else(|| lean_imm(name, o.imm), str::to_string);

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

// Load fault class and alignment-assert exception class for a source kind. The
// alignment assert reports a store/AMO fault for any memory write or atomic and a
// load fault for a plain load. The internal aligned-doubleword `LD` is the read
// side of an atomic (`.amo`) only for true atomics, a normal load otherwise: a
// plain store's read-modify-write read is always 8-aligned, so the class is
// unreachable at runtime (`Semantics.lean` consults it only on a misaligned
// address) and merely selects the Sail exception class the proof layer expects.
fn fault_info(n: &str) -> (&'static str, &'static str) {
    let write_or_atomic = n.starts_with("AMO")
        || n.starts_with("LR")
        || matches!(n, "SB" | "SH" | "SW" | "SD" | "SCD" | "SCW");
    let align_fault = if write_or_atomic {
        "(ExceptionType.E_SAMO_Addr_Align ())"
    } else {
        "(ExceptionType.E_Load_Addr_Align ())"
    };
    let atomic = n.starts_with("AMO") || n.starts_with("LR") || matches!(n, "SCD" | "SCW");
    let load_class = if atomic { "amo" } else { "normal" };
    (load_class, align_fault)
}

// Which source operands an expansion arm actually references, so a generated
// `def` takes exactly those parameters. `advice_count` is how many advice values
// it consumes (each becomes a symbolic `BitVec 64` parameter).
struct ArmInfo {
    uses_rd: bool,
    uses_rs1: bool,
    uses_rs2: bool,
    uses_imm: bool,
    advice_count: usize,
}

// Expand `kind` with sentinel operands (rd, rs1=2, rs2=3, imm=SRC_IMM) and render
// each final row as a Lean `.instr (…) <|` line. `rd` selects the arm: 0 is the
// rd==x0 branch, non-zero the general branch. Operand usage is read from the row
// fields (which sentinels appear), and advice rows — identified by the canonical
// `Advice` circuit flag — get a fresh symbolic `adviceN` immediate.
fn arm_lines(
    kind: SourceInstructionKind,
    rd: u8,
    load_class: &str,
    align_fault: &str,
) -> Result<(Vec<String>, ArmInfo), ExpansionError> {
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
    let mut lines = Vec::new();
    let mut info = ArmInfo {
        uses_rd: false,
        uses_rs1: false,
        uses_rs2: false,
        uses_imm: false,
        advice_count: 0,
    };
    for instruction in expand_instruction(&input, &mut allocator, RV64IMAC_JOLT)? {
        let final_row = JoltInstructionRow::from(instruction);
        let is_advice = JoltInstruction::try_from(final_row)
            .is_ok_and(|i| i.circuit_flags().get(CircuitFlags::Advice));
        for reg in [
            final_row.operands.rd,
            final_row.operands.rs1,
            final_row.operands.rs2,
        ] {
            match reg {
                Some(1) => info.uses_rd = true,
                Some(2) => info.uses_rs1 = true,
                Some(3) => info.uses_rs2 = true,
                _ => {}
            }
        }
        let advice_name = if is_advice {
            let name = format!("advice{}", info.advice_count);
            info.advice_count += 1;
            Some(name)
        } else {
            if final_row.operands.imm == SRC_IMM {
                info.uses_imm = true;
            }
            None
        };
        lines.push(lean_instr(
            &final_row,
            load_class,
            align_fault,
            advice_name.as_deref(),
        ));
    }
    Ok((lines, info))
}

// Expand `kind` and print it as a Lean `Program`, both arms of the rd==x0 branch.
fn emit_program(kind: SourceInstructionKind) -> Result<(), ExpansionError> {
    let n = kind.name();
    let (load_class, align_fault) = fault_info(n);
    println!("--- {n} ---");
    for (arm, rd) in [("rd == x0", 0u8), ("rd != x0", 1u8)] {
        println!("{arm}:");
        let (lines, _) = arm_lines(kind, rd, load_class, align_fault)?;
        for line in lines {
            println!("  {line}");
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

// Print one instruction as a Lean `def …ProgramAuto`. Kinds that write `rd`
// render the rd==x0 branch explicitly as `if isX0 rd then … else …`; stores
// (which never write `rd`) render a single body. Source operands actually
// referenced become `regidx` parameters, and the pass-through immediate a
// `BitVec 12`.
fn emit_lean_def(kind: SourceInstructionKind) -> Result<(), ExpansionError> {
    let n = kind.name();
    let (load_class, align_fault) = fault_info(n);
    let (general, info) = arm_lines(kind, 1, load_class, align_fault)?;

    let mut regs: Vec<&str> = Vec::new();
    if info.uses_rd {
        regs.push("rd");
    }
    if info.uses_rs1 {
        regs.push("rs1");
    }
    if info.uses_rs2 {
        regs.push("rs2");
    }
    let mut params = String::new();
    if !regs.is_empty() {
        params.push('(');
        params.push_str(&regs.join(" "));
        params.push_str(" : regidx)");
    }
    if info.uses_imm {
        if !params.is_empty() {
            params.push(' ');
        }
        params.push_str("(imm : BitVec 12)");
    }
    if info.advice_count > 0 {
        if !params.is_empty() {
            params.push(' ');
        }
        let names: Vec<String> = (0..info.advice_count)
            .map(|i| format!("advice{i}"))
            .collect();
        params.push('(');
        params.push_str(&names.join(" "));
        params.push_str(" : BitVec 64)");
    }

    let def_name = format!("{}ProgramAuto", n.to_lowercase());
    println!("/-- Auto-generated from the Rust `{n}` expansion. -/");
    if params.is_empty() {
        println!("def {def_name} : Program :=");
    } else {
        println!("def {def_name} {params} : Program :=");
    }

    if info.uses_rd {
        let (x0, _) = arm_lines(kind, 0, load_class, align_fault)?;
        println!("  if isX0 rd then");
        for line in x0 {
            println!("    {line}");
        }
        println!("    .done RETIRE_SUCCESS");
        println!("  else");
        for line in &general {
            println!("    {line}");
        }
        println!("    .done RETIRE_SUCCESS");
    } else {
        for line in &general {
            println!("  {line}");
        }
        println!("  .done RETIRE_SUCCESS");
    }
    println!();
    Ok(())
}

// Print a full compilable Lean file of `…ProgramAuto` definitions for every
// supported (expandable) source instruction. Unsupported and native kinds are
// skipped. Emit with `jolt-lean-gen --lean`.
fn emit_lean_defs() -> Result<(), Box<dyn std::error::Error>> {
    println!("import JoltBytecode.JoltISA.Instruction");
    println!("import JoltBytecode.JoltISA.VirtualRegisters");
    println!("import JoltBytecode.JoltISA.Expansions.ALU");
    println!();
    println!("/-!");
    println!("# Auto-generated Jolt-ISA expansion programs");
    println!();
    println!("Generated by `jolt-lean-gen --lean` from the Rust bytecode expander");
    println!("(`expand_instruction`), the source of truth. Do not edit by hand;");
    println!("regenerate instead. Each `…ProgramAuto` mirrors the hand-written");
    println!("`…Program` in `JoltISA/Expansions/`; `if isX0 rd` shows both arms of");
    println!("the rd==x0 branch. Immediate helpers come from `Expansions.ALU`.");
    println!("-/");
    println!();
    println!("open Sail PreSail LeanRV64D.Functions");
    println!();
    println!("namespace JoltISA");
    println!();

    let mut ok = 0usize;
    let mut failed = Vec::new();
    for &kind in SourceInstructionKind::ALL {
        if !matches!(classify(kind), Class::Expand) {
            continue;
        }
        match emit_lean_def(kind) {
            Ok(()) => ok += 1,
            Err(e) => failed.push(format!("{}: {e}", kind.name())),
        }
    }
    println!("end JoltISA");
    println!();
    println!("-- generated {ok} expansion program(s)");
    for f in &failed {
        println!("-- failed: {f}");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args()
        .nth(1)
        .ok_or("usage: gen-lean <INSTRUCTION|--all|--lean>  (e.g. LB)")?;
    if arg == "--all" {
        return emit_all();
    }
    if arg == "--lean" {
        return emit_lean_defs();
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
