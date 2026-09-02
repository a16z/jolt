//! Dory polynomial commitment scheme implementation for the Jolt zkVM.
//!
//! Wraps the [Dory](https://eprint.iacr.org/2020/1274) polynomial commitment
//! scheme for BN254 with transparent setup, logarithmic proof size, and
//! logarithmic verification. Supports streaming commitment and additive
//! homomorphism for batch opening reduction.
//!
//! Implements [`CommitmentScheme`](jolt_openings::CommitmentScheme),
//! [`AdditivelyHomomorphic`](jolt_openings::AdditivelyHomomorphic),
//! [`StreamingCommitment`](jolt_openings::StreamingCommitment), and
//! [`ZkOpeningScheme`](jolt_openings::ZkOpeningScheme) from `jolt-openings`.
//!
//! # Public API
//!
//! - [`DoryScheme`] — implements the four PCS traits. Static methods:
//!   `setup_prover` and `setup_verifier`. Use
//!   [`ZkOpeningScheme::commit_zk`](jolt_openings::ZkOpeningScheme::commit_zk)
//!   for hiding commitments. Also implements
//!   [`DeriveSetup<DoryProverSetup>`](jolt_crypto::DeriveSetup) for
//!   [`PedersenSetup<Bn254G1>`](jolt_crypto::PedersenSetup) (use
//!   `PedersenSetup::derive(&prover_setup, capacity)`).
//! - [`DoryCommitment`] — BN254 pairing target element (GT).
//! - [`DoryProof`] — single opening proof.
//! - [`DoryProverSetup`] / [`DoryVerifierSetup`] — prover and verifier SRS.
//! - [`DoryPartialCommitment`] — intermediate state for streaming commitment.
//! - [`DoryHint`] — row commitments and commitment blind reusable as opening proof hint.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![deny(unsafe_op_in_unsafe_fn)]
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

mod hint_hook;
mod host_tail;
mod routines;
mod routines_hook;
mod scheme;
mod streaming;
mod tier2;
mod transcript;
mod types;
#[cfg(not(target_arch = "wasm32"))]
mod urs_lock;

pub use hint_hook::{install_combine_hints_hook, CombineHintsFn, CombineHintsHookGuard};
pub use host_tail::FastTail;
pub use routines::{JoltG1Routines, JoltG2Routines};
pub use routines_hook::{
    install_routine_hooks, G1ScalarMulAddFn, G2FixedBaseMulFn, G2ScalarMulAddFn, RoutineHooks,
    RoutineHooksGuard,
};
pub use scheme::DoryScheme;
pub use tier2::{multi_miller_affine, one_hot_output_from_rows, DoryTier2Prep, Tier2Accumulator};
pub use types::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryVerifierSetup,
};
