//! Tracer-free instruction and cycle traits for Jolt.

mod jolt_cycle;

pub use jolt_cycle::JoltCycle;
pub use jolt_riscv::instructions;
pub use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags,
    InterleavedBitsMarker, JoltInstruction, JoltInstructions, NUM_CIRCUIT_FLAGS,
    NUM_INSTRUCTION_FLAGS,
};
