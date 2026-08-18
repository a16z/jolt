//! Canonical verifier crate for Jolt proofs.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::integer_division,
    clippy::indexing_slicing,
    clippy::unreachable,
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

pub(crate) mod num {
    //! Provably lossless numeric conversions, centralized so the rest of the
    //! crate stays free of `as` casts (`clippy::as_conversions` is denied
    //! crate-wide; fallible conversions use `TryFrom` at the call site).

    /// Widens `usize` to `u64`.
    #[expect(
        clippy::as_conversions,
        reason = "usize is at most 64 bits on every supported target, so the cast is lossless"
    )]
    pub(crate) fn u64_from_usize(value: usize) -> u64 {
        value as u64
    }

    /// Widens `usize` to `u128`.
    #[cfg(feature = "akita")]
    pub(crate) fn u128_from_usize(value: usize) -> u128 {
        u128::from(u64_from_usize(value))
    }

    /// `value.ilog2()` as `usize`. Panics on zero, exactly as `usize::ilog2`
    /// does; callers pass validated power-of-two dimensions.
    #[expect(
        clippy::as_conversions,
        reason = "an ilog2 result is below usize::BITS and always fits usize"
    )]
    pub(crate) fn ilog2(value: usize) -> usize {
        value.ilog2() as usize
    }
}

pub mod config;
pub mod error;
#[cfg(feature = "fs-audit")]
#[doc(hidden)]
pub mod fs_audit;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{validate_proof_config, JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG};
pub use error::VerifierError;
pub use preprocessing::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
#[cfg(feature = "akita")]
pub use verifier::absorb_packed_commitments;
#[cfg(not(feature = "akita"))]
pub use verifier::absorb_transcript_commitments;
pub use verifier::{
    absorb_committed_program_commitments, absorb_transcript_preamble, validate_and_seed_transcript,
    validate_inputs_from_parts, verify, verify_until_stage1, CheckedInputs, PreStage1VerifierState,
    ProofTranscriptConfig,
};
