//! Source-compatible facade for `jolt-sumcheck`, `jolt-openings`, and
//! `jolt-crypto`.
//!
//! Wraps a duplex sponge over each of the three backends and re-exposes
//! the legacy `Transcript` / `AppendToTranscript` API. Removed once
//! jolt-core migrates to the split-trait surface.

use std::marker::PhantomData;

use jolt_field::{CanonicalBytes, Field, FromPrimitiveInt, TranscriptChallenge};
use spongefish::{DuplexSpongeInterface, Encoding};

use crate::codec::BytesMsg;
use crate::setup::{EmptyInstance, PROTOCOL_ID};

/// Maximum label length in bytes accepted by [`Transcript::new`] and the
/// label helpers below.
pub const MAX_LABEL_LEN: usize = 32;

/// Fiat-Shamir transcript for non-interactive proofs.
///
/// A transcript absorbs data and produces deterministic challenges. Both
/// prover and verifier maintain identical transcripts to derive the same
/// challenges.
///
/// # Security
///
/// The label passed to [`new`](Transcript::new) is mapped to the
/// spongefish session value, so distinct labels carry distinct domain
/// barriers.
pub trait Transcript: Default + Sync + Send + 'static {
    /// The challenge type produced by this transcript.
    type Challenge: TranscriptChallenge;

    /// Creates a new transcript with the given domain separation label.
    ///
    /// # Panics
    ///
    /// Panics if `label.len() > MAX_LABEL_LEN`.
    fn new(label: &'static [u8]) -> Self;

    /// Absorbs raw bytes.
    fn append_bytes(&mut self, bytes: &[u8]);

    /// Absorbs a value via [`AppendToTranscript`].
    fn append<A: AppendToTranscript>(&mut self, value: &A) {
        value.append_to_transcript(self);
    }

    /// Absorbs a domain label followed by a value.
    fn append_labeled<A: AppendToTranscript>(&mut self, label: &'static [u8], value: &A) {
        self.append(&Label(label));
        self.append(value);
    }

    /// Absorbs a domain label with a count followed by each value in order.
    fn append_values<A: AppendToTranscript>(&mut self, label: &'static [u8], values: &[A]) {
        self.append(&LabelWithCount(label, values.len() as u64));
        for value in values {
            self.append(value);
        }
    }

    /// Squeezes a challenge.
    #[must_use]
    fn challenge(&mut self) -> Self::Challenge;

    /// Squeezes a non-optimized scalar challenge from the transcript.
    #[must_use]
    fn challenge_scalar(&mut self) -> Self::Challenge {
        self.challenge()
    }

    /// Squeezes `len` challenges.
    #[must_use]
    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        (0..len).map(|_| self.challenge()).collect()
    }

    /// Squeezes one scalar challenge and returns its powers `[1, gamma, gamma^2, ...]`.
    #[must_use]
    fn challenge_scalar_powers(&mut self, len: usize) -> Vec<Self::Challenge>
    where
        Self::Challenge: Field,
    {
        let gamma = self.challenge_scalar();
        let mut powers = vec![Self::Challenge::from_u64(1); len];
        for index in 1..len {
            powers[index] = powers[index - 1] * gamma;
        }
        powers
    }

    /// Current 256-bit transcript state. Peeked non-destructively by
    /// squeezing 32 bytes from a sponge clone, so callers can read it
    /// without advancing the real state. Useful for debug-only
    /// cross-verifier comparison.
    #[must_use]
    fn state(&self) -> [u8; 32];

    /// Enables transcript comparison for tests; mirrors upstream's signature.
    /// Spongefish sponges have no replayable state history, so this is a
    /// no-op on the legacy facade — call sites already only use it under
    /// `#[cfg(test)]` for debugging digest-based transcripts.
    #[cfg(test)]
    fn compare_to(&mut self, _other: &Self) {}
}

/// Implement on types that absorb themselves into a [`Transcript`].
pub trait AppendToTranscript {
    /// Absorbs this value into the transcript.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);

    /// Byte length of the payload absorbed by [`append_to_transcript`], when
    /// the type participates in jolt-core's variable-length labeled appends.
    fn transcript_payload_len(&self) -> Option<u64> {
        None
    }
}

/// Big-endian field element absorption (matches jolt-core's EVM-compatible
/// byte order).
impl<F: CanonicalBytes> AppendToTranscript for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.to_bytes_le(&mut buf);
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}

/// 32-byte zero-padded label word (matches jolt-core's `raw_append_label`).
pub struct Label(pub &'static [u8]);

impl AppendToTranscript for Label {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 32,
            "label {:?} exceeds 32 bytes",
            core::str::from_utf8(self.0)
        );
        let mut padded = [0u8; 32];
        padded[..self.0.len()].copy_from_slice(self.0);
        transcript.append_bytes(&padded);
    }
}

/// Packed label (24 bytes) + count (8-byte big-endian) in one 32-byte word
/// (matches jolt-core's `raw_append_label_with_len`).
pub struct LabelWithCount(pub &'static [u8], pub u64);

impl AppendToTranscript for LabelWithCount {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 24,
            "label {:?} exceeds 24 bytes",
            core::str::from_utf8(self.0)
        );
        let mut packed = [0u8; 32];
        packed[..self.0.len()].copy_from_slice(self.0);
        packed[24..32].copy_from_slice(&self.1.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

/// EVM-compatible left-padded u64: 24 zero bytes + 8-byte BE value (matches
/// jolt-core's `raw_append_u64`).
pub struct U64Word(pub u64);

impl AppendToTranscript for U64Word {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut packed = [0u8; 32];
        packed[24..].copy_from_slice(&self.0.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

/// Sponge-backed transcript driving a duplex sponge directly.
///
/// The legacy facade does not produce or consume a NARG byte string —
/// existing modular consumers only call `append_bytes` / `challenge` /
/// `state`. New code should use [`crate::ProverTranscript`] /
/// [`crate::VerifierTranscript`] instead.
///
/// Construction mirrors spongefish's `DomainSeparator` builder:
/// `protocol_id || session(label) || instance(())` are absorbed in order.
pub struct SpongeTranscript<H, F = jolt_field::Fr>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: TranscriptChallenge,
{
    sponge: H,
    _field: PhantomData<F>,
}

impl<H, F> Default for SpongeTranscript<H, F>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: TranscriptChallenge,
{
    fn default() -> Self {
        Self::new(b"")
    }
}

fn absorb_encoded<H, T>(sponge: &mut H, value: &T)
where
    H: DuplexSpongeInterface<U = u8>,
    T: Encoding<[u8]> + ?Sized,
{
    let _ = sponge.absorb(value.encode().as_ref());
}

/// Peeks 32 bytes from a clone of the sponge so the real state stays put.
fn peek_state<H: DuplexSpongeInterface<U = u8> + Clone>(sponge: &H) -> [u8; 32] {
    let mut clone = sponge.clone();
    let mut buf = [0u8; 32];
    let _ = clone.squeeze(&mut buf);
    buf
}

impl<H, F> Transcript for SpongeTranscript<H, F>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: TranscriptChallenge,
{
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        assert!(
            label.len() <= MAX_LABEL_LEN,
            "label must be at most {MAX_LABEL_LEN} bytes",
        );
        let mut sponge = H::default();
        absorb_encoded(&mut sponge, &PROTOCOL_ID);
        absorb_encoded(&mut sponge, &BytesMsg(label.to_vec()));
        absorb_encoded(&mut sponge, &EmptyInstance);
        Self {
            sponge,
            _field: PhantomData,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // 1-byte non-zero domain marker + 8-byte LE length + body.
        //
        // The marker sub-domain-separates legacy-facade `append_bytes` calls
        // from spongefish-native `public_message` / `prover_message` calls on
        // the same sponge type, so a future protocol that mixes both paths
        // can't have a legacy append collide with a spongefish-native
        // BytesMsg of the same body. The length prefix keeps
        // `append_bytes(a) ; append_bytes(b)` distinct from
        // `append_bytes(a || b)`.
        const APPEND_MARKER: u8 = 0x9B;
        let mut buf = Vec::with_capacity(9 + bytes.len());
        buf.push(APPEND_MARKER);
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
        let _ = self.sponge.absorb(&buf);
    }

    fn challenge(&mut self) -> F {
        // WARNING: this squeezes 16 bytes for every sponge — including
        // Poseidon — even though the split-trait surface (`OptimizedChallenge`,
        // see `prover.rs:53-55`) deliberately makes 128-bit challenges a
        // compile error on Poseidon-backed states. The two surfaces
        // disagree on purpose: the legacy facade preserves the legacy
        // jolt-core challenge width for in-flight consumers (jolt-sumcheck,
        // jolt-openings, jolt-crypto). Once those migrate to the split-trait
        // surface this facade goes away and the inconsistency with it.
        let mut buf = [0u8; 16];
        let _ = self.sponge.squeeze(&mut buf);
        F::from_challenge_bytes(&buf)
    }

    fn challenge_scalar(&mut self) -> F {
        // Mirrors the digest transcript's scalar challenge: same 16-byte
        // squeeze width as `challenge`, but the non-optimized decoding path.
        let mut buf = [0u8; 16];
        let _ = self.sponge.squeeze(&mut buf);
        F::from_scalar_challenge_bytes(&buf)
    }

    fn state(&self) -> [u8; 32] {
        peek_state(&self.sponge)
    }
}
