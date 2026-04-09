pub mod flags;
pub mod rv;
pub mod traits;
pub mod virt;

pub use flags::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags,
    InterleavedBitsMarker, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
pub use traits::Instruction;
