//! Fiat-Shamir transcripts for Jolt in the NARG model, on spongefish.
//!
//! A proof is its argument string. [`ProverTranscript`] and
//! [`VerifierTranscript`] are typed layers over spongefish's `ProverState` and
//! `VerifierState`: the prover appends every message and absorbs its
//! encoding, the verifier reads it back and absorbs the same encoding. No
//! code reads or copies spongefish's sponge state. Code that both
//! roles run identically is written against [`Channel`]. Protocols compose by
//! sharing one transcript: a sub-protocol takes the caller's transcript rather
//! than starting its own.
//!
//! Message atoms are the [`CanonicalBytes`](jolt_field::CanonicalBytes) /
//! [`CanonicalDecode`](jolt_field::CanonicalDecode) codecs owned by each
//! type's crate, whose spongefish `Encoding` / `NargDeserialize` they are. Challenges are exactly uniform ([`Channel::challenge`]) or
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
mod error;
mod fork;
mod grinding;
mod nonce;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod protocol;
mod prover;
mod site;
mod sponge;
mod state;
mod verifier;

pub use channel::Channel;
pub use error::TranscriptError;
pub use fork::{Fork, FORK_SEED_LEN};
pub use grinding::{
    grinding_predicate_accepts, GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_LEN,
    MAX_GRINDING_BITS,
};
pub use nonce::Nonce;
pub use protocol::ProtocolId;
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
