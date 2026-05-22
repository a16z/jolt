//! Hyrax multilinear polynomial commitment adapter.
//!
//! This crate implements Hyrax as a thin row-commitment layer over
//! [`jolt_crypto::VectorCommitment`]. V1 is transparent: row commitments use
//! zero blinding and the opening hint is `()`.

mod commitment;
mod dimensions;
mod error;
mod proof;
mod scheme;
mod setup;

pub use commitment::HyraxCommitment;
pub use dimensions::HyraxDimensions;
pub use error::HyraxError;
pub use proof::HyraxOpeningProof;
pub use scheme::HyraxScheme;
pub use setup::{HyraxProverSetup, HyraxSetupParams, HyraxVerifierSetup};
