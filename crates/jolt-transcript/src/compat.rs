//! Source-compatible facade for `jolt-sumcheck`, `jolt-openings`, and
//! `jolt-crypto`.
//!
//! Wraps a spongefish `ProverState` over each of the three sponges and
//! re-exposes the legacy `Transcript` / `AppendToTranscript` API. Removed
//! once jolt-core migrates to the split-trait surface.

use std::marker::PhantomData;

use jolt_field::Field;
use spongefish::{DuplexSpongeInterface, Encoding};

use crate::codec::BytesMsg;
use crate::domain::{EmptyInstance, PROTOCOL_ID};

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
pub trait Transcript: Default + Clone + Sync + Send + 'static {
    /// The challenge type produced by this transcript.
    type Challenge: Copy + Default + PartialEq + Eq + std::fmt::Debug + std::hash::Hash;

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

    /// Squeezes a challenge.
    #[must_use]
    fn challenge(&mut self) -> Self::Challenge;

    /// Squeezes `len` challenges.
    #[must_use]
    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        (0..len).map(|_| self.challenge()).collect()
    }
}

/// Implement on types that absorb themselves into a [`Transcript`].
pub trait AppendToTranscript {
    /// Absorbs this value into the transcript.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);
}

/// Big-endian field element absorption (matches jolt-core's EVM-compatible
/// byte order).
impl<F: Field> AppendToTranscript for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = self.to_bytes();
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
/// The compat facade does not produce or consume a NARG byte string —
/// existing modular consumers (jolt-sumcheck, jolt-openings, jolt-crypto)
/// only call `append_bytes` / `challenge`. New code should use
/// [`crate::ProverTranscript`] / [`crate::VerifierTranscript`] instead.
///
/// Construction mirrors spongefish's `DomainSeparator` builder:
/// `protocol_id || session(label) || instance(())` are absorbed in order.
pub struct SpongeTranscript<H, F = jolt_field::Fr>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: Field,
{
    sponge: H,
    _field: PhantomData<F>,
}

impl<H, F> Default for SpongeTranscript<H, F>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: Field,
{
    fn default() -> Self {
        Self::new(b"")
    }
}

impl<H, F> Clone for SpongeTranscript<H, F>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: Field,
{
    fn clone(&self) -> Self {
        Self {
            sponge: self.sponge.clone(),
            _field: PhantomData,
        }
    }
}

fn absorb_encoded<H, T>(sponge: &mut H, value: &T)
where
    H: DuplexSpongeInterface<U = u8>,
    T: Encoding<[u8]> + ?Sized,
{
    let _ = sponge.absorb(value.encode().as_ref());
}

impl<H, F> Transcript for SpongeTranscript<H, F>
where
    H: DuplexSpongeInterface<U = u8> + Clone + Default + Send + Sync + 'static,
    F: Field,
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
        // The marker sub-domain-separates compat-facade `append_bytes` calls
        // from spongefish-native `public_message` / `prover_message` calls on
        // the same sponge type, so a future protocol that mixes both paths
        // can't have a compat append collide with a spongefish-native
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
        let mut buf = [0u8; 16];
        let _ = self.sponge.squeeze(&mut buf);
        F::from_u128(u128::from_le_bytes(buf))
    }
}
