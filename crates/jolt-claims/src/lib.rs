//! Shared claim and expression types for Jolt protocols.

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
mod util;

pub use claim_data::{
    zip_openings, GetPoint, GetValue, InputClaims, OpeningClaim, OutputClaims, ZipOpenings,
};
pub use jolt_claims_derive::{InputClaims, OutputClaims};
pub use claims::{challenge, constant, derived, opening, Expr, Source, Term};
pub use sumcheck::{SumcheckDomain, SumcheckSpec};
pub use symbolic::SymbolicSumcheck;
