pub const XLEN: usize = 64;

pub mod challenge_ops;
pub mod instructions;
pub mod interleave;
pub mod lookup_bits;
pub mod materialize;
pub mod tables;
pub mod traits;

pub use challenge_ops::{ChallengeOps, FieldOps, LookupEval};
pub use interleave::{interleave_bits, uninterleave_bits};
pub use lookup_bits::LookupBits;
pub use materialize::{LookupMaterializer, MaterializerBackend, U128Materializer};
pub use tables::prefixes::ALL_PREFIXES;
pub use tables::{LookupTableKind, PrefixSuffixDecomposition};
pub use traits::{InstructionLookupTable, JoltLookupQuery, LookupQuery, LookupTable};
