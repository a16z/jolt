//! Fiat-Shamir transcripts for Jolt, backed by spongefish.
//!
//! Two surfaces:
//!
//! - **Split spongefish-native traits** ([`ProverTranscript`],
//!   [`VerifierTranscript`], [`OptimizedChallenge`]) — implemented directly
//!   on `spongefish::ProverState` / `spongefish::VerifierState`. Use these
//!   for new code.
//! - **Source-compatible facade** ([`Transcript`], [`AppendToTranscript`],
//!   [`Blake2bTranscript`], [`KeccakTranscript`], [`PoseidonTranscript`]) —
//!   preserved for `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto` while
//!   those crates migrate to the split-trait surface.
//!
//! Three sponges feature-gated: `transcript-blake2b` (spongefish
//! `Blake2b512`), `transcript-keccak` (spongefish `Keccak`),
//! `transcript-poseidon` (local Circom-compatible BN254 [`PoseidonSponge`]).

#![deny(missing_docs)]
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

#[cfg(feature = "spongefish")]
mod codec;
#[cfg(feature = "digest")]
mod digest;
mod legacy;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
#[cfg(feature = "spongefish")]
mod prover;
#[cfg(feature = "spongefish")]
mod setup;
#[cfg(feature = "spongefish")]
mod verifier;

#[cfg(feature = "spongefish")]
pub use codec::BytesMsg;
#[cfg(feature = "digest")]
pub use digest::DigestTranscript;
#[cfg(feature = "spongefish")]
pub use legacy::SpongeTranscript;
pub use legacy::{
    append_length_prefixed, AppendToTranscript, Label, LabelWithCount, Transcript, U64Word,
    MAX_LABEL_LEN,
};
#[cfg(feature = "spongefish")]
pub use setup::{prover_transcript, transcript_builder, verifier_transcript, PROTOCOL_ID};

/// Source-compatible re-exports of legacy label / count / word helpers
/// under their `jolt_transcript::domain::*` path (matches the path used
/// by jolt-dory and earlier modular consumers).
pub mod domain {
    pub use crate::legacy::{Label, LabelWithCount, U64Word};
}

#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonSponge;
#[cfg(all(feature = "bn254", feature = "spongefish"))]
pub use prover::OptimizedChallenge;
#[cfg(feature = "spongefish")]
pub use prover::ProverTranscript;
#[cfg(feature = "spongefish")]
pub use verifier::VerifierTranscript;

#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
use blake2::{digest::consts::U32, Blake2b};
#[cfg(all(
    feature = "bn254",
    any(
        feature = "transcript-blake2b",
        feature = "transcript-keccak",
        feature = "transcript-poseidon"
    )
))]
use jolt_field::Fr;
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
use jolt_inlines_blake2::digest_adapter::{Blake2b, U32, U64};
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
use spongefish::instantiations::hash::Hash;
#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
use spongefish::instantiations::Blake2b512;
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
type Blake2b512 = Hash<Blake2b<U64>>;
#[cfg(feature = "transcript-keccak")]
use spongefish::instantiations::Keccak;

/// Fiat-Shamir transcript backed by Blake2b-512 (spongefish duplex sponge).
#[cfg(all(feature = "transcript-blake2b", feature = "bn254"))]
pub type Blake2bTranscript<F = Fr> = SpongeTranscript<Blake2b512, F>;
/// Fiat-Shamir transcript backed by Blake2b-512 for an explicitly selected field.
#[cfg(all(feature = "transcript-blake2b", not(feature = "bn254")))]
pub type Blake2bTranscript<F> = SpongeTranscript<Blake2b512, F>;

/// Blake2b-256 chained-digest transcript used by the deployed proof format.
/// New protocols should use [`Blake2bTranscript`] instead.
#[cfg(all(feature = "transcript-blake2b", feature = "bn254"))]
pub type LegacyBlake2bTranscript<F = Fr> = DigestTranscript<Blake2b<U32>, F>;
/// Legacy Blake2b-256 transcript for an explicitly selected field.
#[cfg(all(feature = "transcript-blake2b", not(feature = "bn254")))]
pub type LegacyBlake2bTranscript<F> = DigestTranscript<Blake2b<U32>, F>;

/// Fiat-Shamir transcript backed by Keccak-f1600 (spongefish duplex sponge).
#[cfg(all(feature = "transcript-keccak", feature = "bn254"))]
pub type KeccakTranscript<F = Fr> = SpongeTranscript<Keccak, F>;
/// Fiat-Shamir transcript backed by Keccak-f1600 for an explicitly selected field.
#[cfg(all(feature = "transcript-keccak", not(feature = "bn254")))]
pub type KeccakTranscript<F> = SpongeTranscript<Keccak, F>;

/// Fiat-Shamir transcript backed by Circom-compatible BN254 Poseidon.
#[cfg(feature = "transcript-poseidon")]
pub type PoseidonTranscript<F = Fr> = SpongeTranscript<PoseidonSponge, F>;
