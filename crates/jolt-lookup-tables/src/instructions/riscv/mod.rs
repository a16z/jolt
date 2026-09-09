//! Per-instruction impls for RV64I/M base ISA instructions that the prover sees directly.
//!
//! Mirrors `crates/jolt-prover-legacy/src/zkvm/instruction/`. Decomposed instructions
//! (W-suffix, multi-byte loads/stores, plain shifts, MULH/MULHSU, DIV/REM, NOOP)
//! live in tracer as virtual sequences and never reach this layer.

use jolt_riscv::{JoltCycle, JoltInstructionRow};

fn doubleword_memory_instruction_inputs<const XLEN: usize, C: JoltCycle>(cycle: &C) -> (u64, i128) {
    let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
    let instruction: JoltInstructionRow = cycle.instruction().into();
    (
        cycle.rs1_val().unwrap_or(0) & mask,
        instruction.operands.imm,
    )
}

fn doubleword_memory_lookup_operands<const XLEN: usize, C: JoltCycle>(cycle: &C) -> (u64, u128) {
    let (address, offset) = doubleword_memory_instruction_inputs::<XLEN, _>(cycle);
    (0, address.wrapping_add(offset as u64).into())
}

fn doubleword_memory_lookup_output<const XLEN: usize, C: JoltCycle>(cycle: &C) -> u64 {
    doubleword_memory_lookup_operands::<XLEN, _>(cycle)
        .1
        .is_multiple_of(8)
        .into()
}

pub mod add;
pub mod addi;
pub mod addiw;
pub mod addw;
pub mod and;
pub mod andi;
pub mod andn;
pub mod auipc;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod ebreak;
pub mod ecall;
pub mod fence;
pub mod jal;
pub mod jalr;
pub mod ld;
pub mod lui;
pub mod mul;
pub mod mulhu;
pub mod mulw;
pub mod or;
pub mod ori;
pub mod sd;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sub;
pub mod subw;
pub mod xor;
pub mod xori;
