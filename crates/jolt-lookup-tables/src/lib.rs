// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
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
