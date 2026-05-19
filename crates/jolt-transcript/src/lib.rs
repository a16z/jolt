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
//!   be retired once `jolt-core` migrates to the split-trait surface.
//!
//! Three sponges feature-gated: `transcript-blake2b` (spongefish
//! `Blake2b512`), `transcript-keccak` (spongefish `Keccak`),
//! `transcript-poseidon` (local Circom-compatible BN254 [`PoseidonSponge`]).

#![deny(missing_docs)]

mod codec;
mod legacy;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod prover;
mod setup;
mod verifier;

pub use codec::BytesMsg;
pub use legacy::{
    AppendToTranscript, Label, LabelWithCount, SpongeTranscript, Transcript, U64Word, MAX_LABEL_LEN,
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

/// Fiat-Shamir transcript backed by Blake2b-512 (spongefish duplex sponge).
#[cfg(feature = "transcript-blake2b")]
pub type Blake2bTranscript<F = jolt_field::Fr> =
    SpongeTranscript<spongefish::instantiations::Blake2b512, F>;

/// Fiat-Shamir transcript backed by Keccak-f1600 (spongefish duplex sponge).
#[cfg(feature = "transcript-keccak")]
pub type KeccakTranscript<F = jolt_field::Fr> =
    SpongeTranscript<spongefish::instantiations::Keccak, F>;

/// Fiat-Shamir transcript backed by Circom-compatible BN254 Poseidon.
#[cfg(feature = "transcript-poseidon")]
pub type PoseidonTranscript<F = jolt_field::Fr> = SpongeTranscript<PoseidonSponge, F>;
