#![expect(
    clippy::explicit_counter_loop,
    clippy::len_zero,
    clippy::redundant_else,
    clippy::unimplemented,
    clippy::unnecessary_cast,
    clippy::unwrap_used,
    clippy::useless_conversion,
    reason = "donor prefix MLE implementations preserve prover relation semantics; lint cleanup is separate"
)]

pub const XLEN: usize = 64;

pub mod challenge_ops;
pub mod instructions;
pub mod interleave;
pub mod lookup_bits;
pub mod tables;
pub mod traits;

pub use challenge_ops::{ChallengeOps, FieldOps};
pub use interleave::{interleave_bits, uninterleave_bits};
pub use lookup_bits::LookupBits;
pub use tables::prefixes::ALL_PREFIXES;
pub use tables::{LookupTableKind, PrefixSuffixDecomposition};
pub use traits::{InstructionLookupTable, JoltLookupQuery, LookupQuery, LookupTable};
