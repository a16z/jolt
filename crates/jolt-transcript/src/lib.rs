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
//!   preserved for `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto`. Will
//!   be retired once `jolt-prover-legacy` migrates to the split-trait surface.
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

mod codec;
mod digest;
mod legacy;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod prover;
mod setup;
mod verifier;

pub use codec::BytesMsg;
pub use digest::DigestTranscript;
pub use legacy::{
    append_length_prefixed, AppendToTranscript, Label, LabelWithCount, SpongeTranscript,
    Transcript, U64Word, MAX_LABEL_LEN,
};
pub use setup::{prover_transcript, transcript_builder, verifier_transcript, PROTOCOL_ID};

/// Source-compatible re-exports of legacy label / count / word helpers
/// under their `jolt_transcript::domain::*` path (matches the path used
/// by jolt-dory and earlier modular consumers).
pub mod domain {
    pub use crate::legacy::{Label, LabelWithCount, U64Word};
}

#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonSponge;
pub use prover::{OptimizedChallenge, ProverTranscript};
pub use verifier::VerifierTranscript;

#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
use blake2::{digest::consts::U32, Blake2b};
#[cfg(any(
    feature = "transcript-blake2b",
    feature = "transcript-keccak",
    feature = "transcript-poseidon"
))]
use jolt_field::Fr;
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
use jolt_inlines_blake2::digest_adapter::{Blake2b, U32, U64};
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
use spongefish::instantiations::hash::Hash;
#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
use spongefish::instantiations::Blake2b512;
#[cfg(feature = "transcript-keccak")]
use spongefish::instantiations::Keccak;

/// Fiat-Shamir transcript backed by Blake2b-512 (spongefish duplex sponge).
#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
pub type Blake2bTranscript<F = Fr> = SpongeTranscript<Blake2b512, F>;
/// Fiat-Shamir transcript backed by Blake2b-512 (spongefish duplex sponge)
/// over the inline hasher.
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
pub type Blake2bTranscript<F = Fr> = SpongeTranscript<Hash<Blake2b<U64>>, F>;

/// Blake2b-256 chained-digest transcript, byte-compatible with `jolt-prover-legacy`'s
/// `Blake2bTranscript`. Required to verify proofs produced by `jolt-prover-legacy`
/// provers; new modular protocols should use [`Blake2bTranscript`] instead.
#[cfg(feature = "transcript-blake2b")]
pub type LegacyBlake2bTranscript<F = Fr> = DigestTranscript<Blake2b<U32>, F>;

/// Fiat-Shamir transcript backed by Keccak-f1600 (spongefish duplex sponge).
#[cfg(feature = "transcript-keccak")]
pub type KeccakTranscript<F = Fr> = SpongeTranscript<Keccak, F>;

/// Fiat-Shamir transcript backed by Circom-compatible BN254 Poseidon.
#[cfg(feature = "transcript-poseidon")]
pub type PoseidonTranscript<F = Fr> = SpongeTranscript<PoseidonSponge, F>;
