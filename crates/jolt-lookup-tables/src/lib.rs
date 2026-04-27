pub const XLEN: usize = 64;

pub mod challenge_ops;
pub mod cycle_analysis;
pub mod instruction_tables;
pub mod interleave;
pub mod lookup_bits;
pub mod tables;
pub mod traits;

pub use challenge_ops::{ChallengeOps, FieldOps};
pub use cycle_analysis::CycleAnalysis;
pub use instruction_tables::InstructionLookupTable;
pub use interleave::{interleave_bits, uninterleave_bits};
pub use lookup_bits::LookupBits;
pub use tables::prefixes::ALL_PREFIXES;
pub use tables::{LookupTableKind, PrefixSuffixDecomposition};
pub use traits::LookupTable;
