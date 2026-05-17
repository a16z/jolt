//! Shared claim and expression types for Jolt protocols.

mod claims;
mod ops;
pub mod protocols;

pub use claims::{
    challenge, constant, opening, public, ClaimExpression, EvaluationClaim, Expr,
    InputClaimExpression, OutputClaimExpression, Source, Term,
};
