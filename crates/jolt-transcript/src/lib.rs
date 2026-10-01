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
mod duplex;
mod error;
mod grinding;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod preview;
mod protocol;
mod prover;
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

#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonSponge;
