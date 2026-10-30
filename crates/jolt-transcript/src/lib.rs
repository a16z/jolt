//! Fiat-Shamir transcripts for Jolt, backed by spongefish.
//!
//! The prover runs spongefish's `ProverState` and the verifier its
//! `VerifierState`. [`Transcript`] and [`FieldTranscript`] are the operations
//! both sides share: public messages, field challenges through each field's
//! [`ChallengeCodec`], and proof-of-work steps whose nonce the proof carries.
//! [`ProverTranscript`] and [`VerifierTranscript`] add the NARG string.
//! [`prover_transcript`] and [`verifier_transcript`] bind the session and the
//! public statement before the first message.
//!
//! The transcript is positional and word aligned: no message carries a label,
//! and every absorb and squeeze is a whole number of eight-byte words
//! ([`PROTOCOL_ID`] lists the layout).
//!
//! Three sponges are feature-gated: `transcript-keccak` (spongefish `Keccak`,
//! the default), `transcript-blake2b` (spongefish `Blake2b512`) and
//! `transcript-poseidon` (a Circom-compatible BN254 [`PoseidonSponge`]).
//! Field challenges need uniform squeezed bytes ([`UniformBytes`]), which
//! Poseidon does not give, so a Poseidon transcript cannot draw them.

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

mod challenge;
mod codec;
mod digest_hash;
mod grind_bits;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod pow;
mod prover;
#[cfg(feature = "test-utils")]
mod scripted;
mod setup;
mod shake128;
mod transcript;
mod verifier;

pub use challenge::{ChallengeCodec, UniformBytes, MAX_CHALLENGE_BYTES};
pub use codec::{protocol_tag, word_padding, BytesMsg, NargMessage, TAG_BYTES, WORD_BYTES};
pub use digest_hash::{DigestHash, DigestHasher, DIGEST_HASH};
pub use grind_bits::{
    grind_bits_for_loss, require_no_pow, ChallengeSetSize, GrindBitsError, MAX_POW_BITS,
    SECURITY_BITS,
};
#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonSponge;
pub use prover::{ProverInit, ProverTranscript};
#[cfg(feature = "test-utils")]
pub use scripted::{Script, ScriptedTranscript, TranscriptEvent};
pub use setup::{prover_transcript, session_id, verifier_transcript, PROTOCOL_ID};
pub use shake128::{ProverState, Shake128, Shake128Hasher};
pub use spongefish::{Encoding, VerificationError, VerifierState};
pub use transcript::{FieldTranscript, Transcript};
pub use verifier::{VerifierInit, VerifierTranscript};

#[cfg(all(
    feature = "transcript-blake2b",
    not(all(feature = "blake2-inline", target_arch = "riscv64"))
))]
use blake2::Blake2b512 as Blake2b512Digest;
#[cfg(all(
    feature = "transcript-blake2b",
    feature = "blake2-inline",
    target_arch = "riscv64"
))]
use jolt_inlines_blake2::digest_adapter::{Blake2b, U64};
#[cfg(all(
    feature = "transcript-blake2b",
    feature = "blake2-inline",
    target_arch = "riscv64"
))]
type Blake2b512Digest = Blake2b<U64>;
#[cfg(feature = "transcript-blake2b")]
use spongefish::instantiations::Hash;

/// Spongefish's fixed-output hash bridge instantiated with Blake2b-512.
#[cfg(feature = "transcript-blake2b")]
pub type Blake2b512 = Hash<Blake2b512Digest>;

/// Spongefish's Keccak-f[1600] duplex sponge.
#[cfg(feature = "transcript-keccak")]
pub use spongefish::instantiations::Keccak;
