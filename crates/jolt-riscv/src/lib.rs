#[macro_use]
mod macros;

pub mod flags;
pub mod instruction_set;
pub mod rv;
pub mod traits;
pub mod virt;

pub use flags::{
    CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS,
    NUM_INSTRUCTION_FLAGS,
};
pub use instruction_set::JoltInstructionSet;
pub use traits::Instruction;

// Re-export jolt-lookup-tables types for downstream convenience
// and for $crate:: paths in define_instruction! macro
pub use jolt_lookup_tables::tables;
pub use jolt_lookup_tables::{
    interleave_bits, uninterleave_bits, ChallengeOps, FieldOps, LookupBits, LookupTable,
    LookupTableKind, LookupTables, PrefixSuffixDecomposition, ALL_PREFIXES,
};
