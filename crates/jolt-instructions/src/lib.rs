//! RISC-V instruction set definitions and lookup table decompositions
//! for the Jolt zkVM proving system.
//!
//! This crate provides:
//!
//! - The [`Instruction`] trait defining the interface for all RISC-V instructions.
//! - The [`LookupTable`] trait for small-domain lookup tables used by the prover.
//! - Concrete implementations of all RV64IMAC instructions with correct `execute` semantics.
//! - Virtual instructions (`ASSERT_EQ`, `ASSERT_LTE`, `POW2`, `MOVSIGN`) used by the VM.
//! - The [`JoltInstructionSet`] registry for opcode-indexed dispatch.
//!
//! # Architecture
//!
//! Each instruction is a zero-sized unit struct implementing [`Instruction`].
//! The `execute` method provides ground-truth computation using native Rust
//! wrapping arithmetic. The `lookups` method will decompose the computation
//! into small lookup table queries when the prover pipeline is integrated.

#[macro_use]
mod macros;

pub mod instruction_set;
pub mod opcodes;
pub mod rv;
pub mod tables;
pub mod traits;
pub mod virtual_;

pub use instruction_set::JoltInstructionSet;
pub use traits::{Instruction, LookupQuery, LookupTable, TableId};
