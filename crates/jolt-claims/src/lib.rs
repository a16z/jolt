//! Shared claim and expression types for Jolt protocols.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes
)]
// wildcard_enum_match_arm is omitted: claim resolvers and test evaluation maps
// are fail-closed by construction (unmatched ids return Err/None/zero, never a
// wrong success path), so the lint would be annotation churn with no catch.

// The per-relation claim structs in `protocols/jolt/relations/**` carry
// `#[derive(InputClaims/OutputClaims)]`, whose generated impls reference the
// claim-data traits and id types through absolute `::jolt_claims::*` paths. Inside
// this crate those paths only resolve via this self-alias.
extern crate self as jolt_claims;

mod claim_data;
mod claims;
mod ops;
pub mod protocols;
mod sumcheck;
mod symbolic;

pub use claim_data::{
    ChallengeDrawError, InputClaims, MissingOpeningValue, NoChallenges, NoInputs, NoOutputs,
    OutputClaims, SumcheckChallenges,
};
pub use claims::{challenge, constant, derived, opening, Expr, Source, Term};
pub use jolt_claims_derive::{InputClaims, OutputClaims, SumcheckChallenges};
pub use sumcheck::SumcheckDomain;
pub use symbolic::SymbolicSumcheck;
