//! Fiat-Shamir transcripts for Jolt, backed by spongefish.
//!
//! The single public surface is the **split spongefish-native traits**
//! ([`ProverTranscript`], [`VerifierTranscript`], [`OptimizedChallenge`]) —
//! implemented directly on `spongefish::ProverState` / `spongefish::VerifierState` —
//! plus the field-typed [`FsTranscript`] / [`FsAbsorb`] / [`FsChallenge`]
//! vocabulary the modular consumer crates bind against.
//!
//! Three sponges feature-gated: `transcript-blake2b` (spongefish
//! `Blake2b512`), `transcript-keccak` (spongefish `Keccak`),
//! `transcript-poseidon` (local Circom-compatible BN254 [`PoseidonSponge`]).

#![deny(missing_docs)]

mod codec;
mod messages;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod prover;
mod setup;
mod verifier;

pub use codec::BytesMsg;
pub use messages::{
    deserialize_slice, serialize_slice, FsAbsorb, FsChallenge, FsNargRead, FsTranscript,
};
pub use setup::{prover_transcript, transcript_builder, verifier_transcript, PROTOCOL_ID};

#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonSponge;
pub use prover::{OptimizedChallenge, ProverTranscript};
pub use verifier::VerifierTranscript;

// Re-export the spongefish state types + sponge interface the split traits are built on,
// so jolt-core and the modular consumers name the entire transcript surface through
// `jolt_transcript` without a direct `spongefish` dependency.
/// Blake2b-512 spongefish sponge instantiation.
#[cfg(feature = "transcript-blake2b")]
pub use spongefish::instantiations::Blake2b512;
/// Keccak-f1600 spongefish sponge instantiation.
#[cfg(feature = "transcript-keccak")]
pub use spongefish::instantiations::Keccak;
/// Spongefish duplex-sponge interface — the `H` sponge parameter of the split traits.
pub use spongefish::DuplexSpongeInterface;
/// Spongefish prover state (`ProverState<H, R>`) — the NARG-emitting transcript.
pub use spongefish::ProverState;
/// Spongefish verifier state (`VerifierState<'a, H>`) — the NARG-reading transcript.
pub use spongefish::VerifierState;
/// Spongefish message-codec traits + NARG (de)serialization + verification result
/// types, re-exported so jolt-core and the modular consumers can author their own
/// `Encoding`/`NargSerialize`/`NargDeserialize` bridges (e.g. for generic field
/// elements and `CanonicalSerialize` blobs) without a direct `spongefish` dependency.
pub use spongefish::{
    Codec, Decoding, Encoding, NargDeserialize, NargSerialize, VerificationError,
    VerificationResult,
};
