//! Akita PCS adapter for Jolt.
//!
//! Wraps the upstream Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` trait plus an adapter delegating same-point batches to
//! Akita's native batched opening protocol.

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

mod adapters;
pub mod configs;
mod native_batching;
pub mod schedules;
mod scheme;
mod shape_guard;

pub use adapters::{
    AkitaBackendFlavor, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_ONE_HOT_K16,
    AKITA_ONE_HOT_K256,
};
pub use native_batching::{
    AkitaNativeBatchPolynomials, AkitaNativeBatchStatement, AkitaNativeBatching,
};
pub use scheme::AkitaScheme;

/// Jolt↔Akita basis-order bridging, exposed so benchmarks measuring the raw
/// backend use the exact transform the adapter uses.
#[doc(hidden)]
pub use adapters::{jolt_to_akita_evals, jolt_to_akita_index, reverse_point};
