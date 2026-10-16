//! Clear binary HyperKZG over BN254, implementing the ordinary opening API.
//!
//! The scheme owns the complete opening transcript prefix. Setup imports
//! authenticated ceremony powers; it never generates a toxic-waste scalar.
//! See the crate README for the statement, source map, and security boundaries.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

mod kzg;
mod scheme;
mod types;

pub use scheme::HyperKZGScheme;
pub use types::{
    HyperKZGError, HyperKZGProof, HyperKZGProverSetup, HyperKZGSetupParams, HyperKZGVerifierSetup,
};
