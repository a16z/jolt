//! The shared masking front end (Whiteout §3.2 and §4.1;
//! `specs/whiteout-masking.md`).
//!
//! Every private field message goes out as m̂ = m + p, with the pads p
//! committed before the first challenge. The verifier records each masked
//! value with its pad slot and, instead of checking it, lowers its checks
//! into the [`ResidualRelation`], which a [`ZkBackend`] proves over the
//! committed pads.

mod backend;
mod lowering;
mod pads;
mod recorder;
mod residual;
#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures"
)]
mod tests;

pub use backend::{ErasedSource, ZkBackend, ZkError};
pub use lowering::{
    lower_factored, sumcheck_rows, ClaimDomain, OperandShapeDomain, RoundFormat, RowsDomain,
    ValuesDomain,
};
pub use pads::{CommittedPads, CommittedPadsView, MaskedVerifier, PadIndex, PadLayout};
pub use recorder::MaskingRecorder;
pub use residual::{
    AffineRow, MaskedOperand, ProductRow, ResidualRelation, ResidualRows, ResidualVar, Row,
};
