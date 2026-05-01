//! Per-instruction impls for RV64I/M base ISA instructions that the prover sees directly.
//!
//! Mirrors `jolt-core/src/zkvm/instruction/`. Decomposed instructions
//! (W-suffix, multi-byte loads/stores, plain shifts, MULH/MULHSU, DIV/REM, NOOP)
//! live in tracer as virtual sequences and never reach this layer.

pub mod add;
pub mod addi;
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
pub mod or;
pub mod ori;
pub mod sd;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sub;
pub mod xor;
pub mod xori;
