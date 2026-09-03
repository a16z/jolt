mod copy_link;
mod scalar_link;
mod terms;

use thiserror::Error;

pub use crate::stream::{AffineForm, ColumnId, Term, TermContext, TermExporter};
pub use copy_link::{CopyLink, CopyLinkClaims, CopyLinkProver, CopyLinkSide, CopyLinkWitness};
pub use scalar_link::{DoryScalarLink, DoryScalarLinkProver};
pub use terms::{
    evaluate_terms, evaluate_terms_observed, CopyLinkTermExporter, CopyLinkTermSide,
    CopyLinkTermsContext, DoryScalarTermExporter, DoryScalarTermsContext, COPY_LINK_TERM_COUNT,
    DORY_SCALAR_TERM_COUNT, MAX_FACTORS,
};

pub const WIRES: usize = 3;
pub const DEGREE: usize = 5;

#[derive(Debug, Error)]
pub enum LinkError {
    #[error("common row domain must be a power of two and hold {minimum} rows, got {actual}")]
    RowDomain { minimum: usize, actual: usize },
    #[error("copy denominator is zero")]
    ZeroDenominator,
    #[error("copy relation is not satisfied")]
    Copy,
    #[error("claim count mismatch")]
    Claims,
}
