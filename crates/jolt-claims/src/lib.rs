//! Shared claim and expression types for Jolt protocols.

mod claims;
mod ops;
pub mod protocols;

pub use claims::{
    challenge, constant, opening, pow2, public, ClaimExpression, ConsistencyClaim, Expr,
    InputClaimExpression, OutputClaimExpression, SameEvaluation, SameEvaluationAs, Source, Term,
};
