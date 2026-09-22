//! Clear binary HyperKZG over BN254, implementing the ordinary opening API.
//!
//! The scheme owns the complete opening transcript prefix. Setup imports
//! authenticated ceremony powers; it never generates a toxic-waste scalar.
//! See the crate README for the statement, source map, and security boundaries.

#![forbid(unsafe_code)]
#![deny(clippy::indexing_slicing, clippy::panic_in_result_fn)]

mod kzg;
mod scheme;
mod types;

pub use scheme::HyperKZGScheme;
pub use types::{
    HyperKZGError, HyperKZGProof, HyperKZGProverSetup, HyperKZGSetupBinding, HyperKZGSetupParams,
    HyperKZGVerifierSetup,
};
