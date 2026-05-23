//! Shared claim and expression types for Jolt protocols.

mod claims;
mod ops;
pub mod protocols;
mod util;

pub use claims::{
    challenge, constant, opening, public, ClaimExpression, ConsistencyClaim, Expr,
    InputClaimExpression, OutputClaimExpression, Source, Term,
};
