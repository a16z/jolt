//! RISC-V instruction set definitions and lookup table decompositions
//! for the Jolt zkVM proving system.
//!
//! This crate provides:
//!
//! - The [`Instruction`] trait: execution semantics, lookup table association, and flags.
//! - The [`LookupTable`] trait: table materialization and MLE evaluation.
//! - The [`Flags`] trait with [`CircuitFlags`] and [`InstructionFlags`] enums.
//! - Concrete implementations of all RV64IMAC + virtual instructions.
//! - The [`JoltInstructionSet`] registry for opcode-indexed dispatch.
//! - Prefix/suffix sparse-dense decomposition for sub-linear MLE evaluation.
//! - Bit-interleaving utilities for two-operand lookup indices.
//!
//! # Architecture
//!
//! Each instruction is a zero-sized unit struct implementing [`Instruction`]
//! (which requires [`Flags`]). The `execute` method provides ground-truth
//! computation. The `lookup_table` method declares which [`LookupTableKind`]
//! the instruction decomposes into for the proving system.
//!
//! Flags are split into *static* (determined by instruction type) and *dynamic*
//! (determined per-cycle by the runtime). This crate provides static flags;
//! dynamic flags (`VirtualInstruction`, `IsCompressed`, `IsRdNotZero`, etc.)
//! are applied by `jolt-zkvm` based on trace context.

#[macro_use]
mod macros;

pub mod challenge_ops;
pub mod flags;
pub mod instruction_set;
pub mod interleave;
pub mod lookup_bits;
pub mod opcodes;
pub mod rv;
pub mod tables;
pub mod traits;
pub mod virtual_;

pub use challenge_ops::{ChallengeOps, FieldOps};
pub use flags::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS,
    NUM_INSTRUCTION_FLAGS,
};
pub use instruction_set::JoltInstructionSet;
pub use interleave::{interleave_bits, uninterleave_bits};
pub use lookup_bits::LookupBits;
pub use tables::LookupTableKind;
pub use traits::{Instruction, LookupTable};
