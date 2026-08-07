//! Per-instruction impls for RV64I/M base ISA instructions that the prover sees directly.
//!
//! Mirrors `crates/jolt-prover-legacy/src/zkvm/instruction/`. Decomposed instructions
//! (W-suffix, multi-byte loads/stores, plain shifts, MULH/MULHSU, DIV/REM, NOOP)
//! live in tracer as virtual sequences and never reach this layer.

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

#[inline]
pub(in crate::instructions) fn sign_extend_half_word<const XLEN: usize>(value: u64) -> u64 {
    let half_word_size = XLEN / 2;
    let lower_mask = (1u128 << half_word_size).wrapping_sub(1) as u64;
    let lower_half = value & lower_mask;
    if lower_half & (1 << (half_word_size - 1)) == 0 {
        lower_half
    } else {
        lower_half | (lower_mask << half_word_size)
    }
}
