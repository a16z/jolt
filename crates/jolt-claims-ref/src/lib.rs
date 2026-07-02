//! Shared claim and expression types for Jolt protocols.
//!
//! This crate owns protocol semantics: symbolic claim expressions, typed opening
//! and challenge identifiers, relation metadata, and Jolt-specific formula
//! builders. The lattice/Akita surface follows the same rule: `jolt-claims`
//! names Jolt facts, packing layout descriptors, logical view formulas, and
//! validity requirements, while PCS transport and witness materialization remain
//! in `jolt-openings`, `jolt-verifier`, and the backend adapter crates.

mod claims;
mod ops;
pub mod protocols;
mod util;

pub use claims::{
    challenge, constant, opening, public, ClaimExpression, ConsistencyClaim, Expr,
    InputClaimExpression, OutputClaimExpression, Source, Term,
};
