//! Source-compatible facade for `jolt-sumcheck`, `jolt-openings`, and
//! `jolt-crypto`.
//!
//! Wraps a duplex sponge over each of the three backends and re-exposes
//! the compatibility `Transcript` / `AppendToTranscript` API. It can be
//! removed once all consumers migrate to the split-trait surface.

#[cfg(feature = "spongefish")]
use std::marker::PhantomData;

#[cfg(feature = "spongefish")]
use jolt_field::CanonicalEncoding;
use jolt_field::{CanonicalBytes, Ring};
#[cfg(feature = "spongefish")]
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, Encoding};

#[cfg(feature = "spongefish")]
use crate::codec::BytesMsg;
#[cfg(feature = "transcript-poseidon")]
use crate::poseidon::{FrUnit, PoseidonSponge};
#[cfg(feature = "spongefish")]
use crate::setup::session_id;
#[cfg(feature = "transcript-poseidon")]
use jolt_field::Fr;

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
/// The label passed to [`new`](Transcript::new) binds the transcript to
/// a protocol domain.
pub trait Transcript: Default + Sync + Send + 'static {
    /// The challenge type produced by this transcript. Concrete transcript
    /// implementations state any decoding or canonical-encoding capability
    /// they need; custom transcripts may return an algebra-only field type.
    type Challenge;

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
        Self::Challenge: Ring,
    {
        let gamma = self.challenge_scalar();
        let one = Self::Challenge::from_u64(1);
        std::iter::successors(Some(one), |power| Some(*power * gamma))
            .take(len)
            .collect()
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

    /// Byte length of the payload absorbed by [`Self::append_to_transcript`], when
    /// the type participates in variable-length labeled appends.
    fn transcript_payload_len(&self) -> Option<u64> {
        None
    }
}

/// Big-endian field element absorption used by the deployed proof format.
impl<F: CanonicalBytes> AppendToTranscript for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.to_bytes_le(&mut buf);
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}

/// Absorbs `label` (with the payload's transcript length when it exposes
/// one) followed by the payload itself. The single home of the labeled-append
/// convention shared by the prover's and verifier's commitment absorption —
/// order and labels are consensus-critical.
pub fn append_length_prefixed<T, A>(transcript: &mut T, label: &'static [u8], payload: &A)
where
    T: Transcript,
    A: AppendToTranscript,
{
    if let Some(len) = payload.transcript_payload_len() {
        transcript.append(&LabelWithCount(label, len));
    } else {
        transcript.append(&Label(label));
    }
    transcript.append(payload);
}

/// 32-byte zero-padded label word used by the deployed proof format.
pub struct Label(pub &'static [u8]);

impl AppendToTranscript for Label {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 32,
            "label {:?} exceeds 32 bytes",
            core::str::from_utf8(self.0)
        );
        let mut padded = [0u8; 32];
        let (head, _) = padded.split_at_mut(self.0.len());
        head.copy_from_slice(self.0);
        transcript.append_bytes(&padded);
    }
}

/// Packed label (24 bytes) + count (8-byte big-endian) in one 32-byte word
/// used by the deployed proof format.
pub struct LabelWithCount(pub &'static [u8], pub u64);

impl AppendToTranscript for LabelWithCount {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 24,
            "label {:?} exceeds 24 bytes",
            core::str::from_utf8(self.0)
        );
        let mut packed = [0u8; 32];
        let (head, _) = packed.split_at_mut(self.0.len());
        head.copy_from_slice(self.0);
        packed[24..32].copy_from_slice(&self.1.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

/// EVM-compatible left-padded u64: 24 zero bytes + 8-byte BE value.
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
/// Construction initializes the protocol session and absorbs the
/// length-prefixed label as a non-empty instance.
#[cfg(feature = "spongefish")]
pub struct SpongeTranscript<H, F>
where
    H: DuplexSpongeInit<U = u8> + Clone + Default + Send + Sync + 'static,
    F: CanonicalEncoding,
{
    sponge: H,
    _field: PhantomData<F>,
}

#[cfg(feature = "spongefish")]
impl<H, F> Default for SpongeTranscript<H, F>
where
    H: DuplexSpongeInit<U = u8> + Clone + Default + Send + Sync + 'static,
    F: CanonicalEncoding,
{
    fn default() -> Self {
        Self::new(b"")
    }
}

/// Rejects labels longer than [`MAX_LABEL_LEN`].
#[cfg(feature = "spongefish")]
fn check_label(label: &[u8]) {
    assert!(
        label.len() <= MAX_LABEL_LEN,
        "label must be at most {MAX_LABEL_LEN} bytes",
    );
}

/// The byte string a legacy `append_bytes` absorbs: a 1-byte non-zero domain
/// marker, the 8-byte LE length, then the body.
///
/// The marker sub-domain-separates legacy-facade `append_bytes` calls from
/// spongefish-native `public_message` / `prover_message` calls on the same
/// sponge type, so a future protocol that mixes both paths can't have a
/// legacy append collide with a spongefish-native BytesMsg of the same body.
/// The length prefix keeps `append_bytes(a) ; append_bytes(b)` distinct from
/// `append_bytes(a || b)`.
#[cfg(feature = "spongefish")]
fn framed_append(bytes: &[u8]) -> Vec<u8> {
    const APPEND_MARKER: u8 = 0x9B;
    let mut buf = Vec::with_capacity(9 + bytes.len());
    buf.push(APPEND_MARKER);
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
    buf
}

/// Peeks 32 bytes from a clone of the sponge so the real state stays put.
#[cfg(feature = "spongefish")]
fn peek_state<H: DuplexSpongeInterface<U = u8> + Clone>(sponge: &H) -> [u8; 32] {
    let mut clone = sponge.clone();
    let mut buf = [0u8; 32];
    let _ = clone.squeeze(&mut buf);
    buf
}

#[cfg(feature = "spongefish")]
impl<H, F> Transcript for SpongeTranscript<H, F>
where
    H: DuplexSpongeInit<U = u8> + Clone + Default + Send + Sync + 'static,
    F: CanonicalEncoding,
{
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        check_label(label);
        let mut sponge = H::init(session_id().as_bytes());
        let _ = sponge.absorb(Encoding::<u8>::encode(&BytesMsg(label.to_vec())).as_ref());
        Self {
            sponge,
            _field: PhantomData,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        let _ = self.sponge.absorb(&framed_append(bytes));
    }

    fn challenge(&mut self) -> F {
        // Squeezes 16 bytes: the legacy deployed challenge width, kept for
        // in-flight consumers (jolt-sumcheck, jolt-openings, jolt-crypto)
        // until they migrate to the split-trait surface.
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

/// Legacy facade over [`PoseidonSponge`], with native BN254 `Fr` challenges.
///
/// Construction initializes the protocol session and absorbs the
/// length-prefixed label, as [`SpongeTranscript`] does. `append_bytes`
/// absorbs the same framed byte string as [`SpongeTranscript`], packed into
/// field elements. Every challenge, scalar or not, is one squeezed element,
/// so challenges are full-width and uniform; there is no 128-bit form.
#[cfg(feature = "transcript-poseidon")]
#[derive(Clone)]
pub struct PoseidonTranscript {
    sponge: PoseidonSponge,
}

#[cfg(feature = "transcript-poseidon")]
impl Default for PoseidonTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

#[cfg(feature = "transcript-poseidon")]
impl PoseidonTranscript {
    fn absorb_bytes(&mut self, bytes: Vec<u8>) {
        let _ = self
            .sponge
            .absorb(<BytesMsg as Encoding<FrUnit>>::encode(&BytesMsg(bytes)).as_ref());
    }
}

#[cfg(feature = "transcript-poseidon")]
impl Transcript for PoseidonTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        check_label(label);
        let mut transcript = Self {
            sponge: PoseidonSponge::init(session_id().as_bytes()),
        };
        transcript.absorb_bytes(label.to_vec());
        transcript
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.absorb_bytes(framed_append(bytes));
    }

    fn challenge(&mut self) -> Fr {
        let [unit] = self.sponge.squeeze_array::<1>();
        unit.into()
    }

    /// The little-endian bytes of the next element, squeezed from a clone.
    fn state(&self) -> [u8; 32] {
        let [unit] = self.sponge.clone().squeeze_array::<1>();
        let mut out = [0u8; 32];
        Fr::from(unit).to_bytes_le(&mut out);
        out
    }
}
