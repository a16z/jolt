//! Shared claim and expression types for Jolt protocols.

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
pub use claims::{challenge, constant, derived, opening, Expr, Source, Term};
pub use sumcheck::{SumcheckDomain, SumcheckSpec};
pub use symbolic::SymbolicSumcheck;
