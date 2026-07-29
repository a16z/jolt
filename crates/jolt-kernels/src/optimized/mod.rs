//! Optimized kernels: the performance tier behind the same [`PrepareKernel`]
//! slots the reference tier serves, byte-parity-tested against it per
//! relation (identical round polynomials and output claims under identical
//! inputs and challenges).
//!
//! [`PrepareKernel`]: crate::PrepareKernel

pub mod registers_claim_reduction;
pub mod registers_read_write;
pub mod registers_val_evaluation;
