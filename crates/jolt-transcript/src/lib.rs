//! Fiat-Shamir transcripts for Jolt in the NARG model.
//!
//! A proof is its argument string. [`ProverTranscript`] appends every prover
//! message and absorbs exactly the appended bytes; [`VerifierTranscript`]
//! reads messages back and absorbs exactly the bytes it read. Code that both
//! roles run identically is written against [`Channel`]. Protocols compose by
//! sharing one transcript: a sub-protocol takes the caller's transcript rather
//! than starting its own.
//!
//! Message atoms are the [`CanonicalBytes`](jolt_field::CanonicalBytes) /
//! [`CanonicalDecode`](jolt_field::CanonicalDecode) codecs owned by each
//! type's crate. Challenges are exactly uniform ([`Channel::challenge`]) or
//! drawn from a field's small challenge set ([`Channel::challenge_small`]).
//! The sponge is a type parameter ([`Sponge`]) bound into the [`ProtocolId`].
//!
//! The legacy symmetric facade ([`Transcript`], [`AppendToTranscript`],
//! [`DigestTranscript`]) remains until every consumer moves to the channel.

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

mod channel;
#[cfg(feature = "spongefish")]
mod codec;
#[cfg(feature = "digest")]
mod digest;
mod duplex;
mod error;
mod grinding;
mod legacy;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod preview;
mod protocol;
mod prover;
#[cfg(feature = "spongefish")]
mod setup;
mod site;
mod sponge;
mod verifier;

pub use channel::Channel;
pub use duplex::SMALL_CHALLENGE_BYTES;
pub use error::TranscriptError;
pub use grinding::{
    grinding_predicate_accepts, GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_LEN,
    MAX_GRINDING_BITS,
};
pub use preview::Preview;
pub use protocol::{ProtocolId, PROTOCOL_ID_LEN};
pub use prover::ProverTranscript;
#[cfg(feature = "logging")]
pub use site::TranscriptEvent;
pub use site::{SiteId, TranscriptOp};
pub use sponge::Sponge;
#[cfg(feature = "transcript-blake2b")]
pub use spongefish::instantiations::Blake2b512;
#[cfg(feature = "transcript-keccak")]
pub use spongefish::instantiations::Keccak;
/// The duplex interface every [`Sponge`] implements; re-exported so a custom
/// sponge needs no direct spongefish dependency.
pub use spongefish::DuplexSpongeInterface;
pub use verifier::VerifierTranscript;

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

#[cfg(feature = "transcript-blake2b")]
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
