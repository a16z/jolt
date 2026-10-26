//! Source-compatible facade for `jolt-sumcheck`, `jolt-openings`, and
//! `jolt-crypto`.
//!
//! Wraps a duplex sponge over each of the three backends and re-exposes
//! the compatibility `Transcript` / `AppendToTranscript` API. It can be
//! removed once all consumers migrate to the split-trait surface.

#[cfg(feature = "spongefish")]
use std::marker::PhantomData;

use jolt_field::{CanonicalBytes, CanonicalEncoding, Ring};
#[cfg(feature = "spongefish")]
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, Encoding};

#[cfg(feature = "spongefish")]
use crate::codec::BytesMsg;
use crate::pow::{Leb128, Probe, ProofOfWorkError, SpongePow};
#[cfg(feature = "spongefish")]
use crate::setup::session_id;
use crate::{challenge_bits_for_loss, ChallengeSetSize, GrindBitsError, SECURITY_BITS};

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
///
/// `Clone` lets the prover search a proof-of-work nonce on copies of the
/// transcript ([`grind`](Transcript::grind)).
pub trait Transcript: Clone + Default + Sync + Send + 'static {
    /// The challenge type produced by this transcript. Concrete transcript
    /// implementations state any decoding or canonical-encoding capability
    /// they need; custom transcripts may return an algebra-only field type.
    type Challenge;

    /// Size of the set every challenge is drawn from, near-uniformly: the
    /// field's [`CHALLENGE_SET_LIMBS`](jolt_field::CanonicalEncoding::CHALLENGE_SET_LIMBS)
    /// for the field transcripts. Prices the proof of work before a challenge
    /// ([`grind_bits_for_loss`]).
    ///
    /// [`grind_bits_for_loss`]: crate::grind_bits_for_loss
    const CHALLENGE_SET: ChallengeSetSize;

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

    /// Squeezes a scalar challenge, such as a batching coefficient. It is
    /// decoded exactly as [`challenge`](Self::challenge) is.
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

    /// Squeezes a scalar challenge on the integers `[0, 2^b)`, for a
    /// step whose bad set holds at most `loss_factor` challenges: `b` is
    /// [`challenge_bits_for_loss`] at [`SECURITY_BITS`], the narrowest width
    /// that needs no proof of work. A group scalar this short costs about
    /// half as much as a full-width field element in BN254.
    ///
    /// It squeezes `⌈b / 8⌉` bytes, clears the bits above `b`, and decodes
    /// the little-endian integer. As `b` is below the bit length of the field
    /// order, distinct integers decode to distinct elements. Nothing about
    /// the field challenges drawn after it changes.
    ///
    /// The integer is uniform on the byte sponges (Keccak, Blake2b). On
    /// Poseidon the bytes are the low bits of a field element, so each value
    /// has probability at most `2^-b (1 + 2^b / p)` when the draw follows an
    /// absorb and so starts a fresh element; a draw that starts mid-element
    /// can read the element's top byte and lose a few bits. Callers draw
    /// short scalars right after an absorb.
    ///
    /// # Errors
    /// [`GrindBitsError`] from [`challenge_bits_for_loss`], or
    /// [`GrindBitsError::FieldTooSmall`] if `b` is not below the bit length
    /// of the field order. Nothing is squeezed on error.
    fn challenge_short_scalar(
        &mut self,
        loss_factor: u64,
    ) -> Result<Self::Challenge, GrindBitsError>
    where
        Self::Challenge: CanonicalEncoding,
    {
        let bits = challenge_bits_for_loss(loss_factor, SECURITY_BITS)?;
        if bits >= Self::Challenge::MODULUS_BITS {
            return Err(GrindBitsError::FieldTooSmall);
        }
        // `bits ≤ 256`, so the bytes fit one 32-byte word.
        let len = bits.div_ceil(8);
        let mut word = [0u8; 32];
        let bytes = word
            .get_mut(..len as usize)
            .ok_or(GrindBitsError::FieldTooSmall)?;
        self.challenge_bytes(bytes);
        if let Some(top) = bytes.last_mut() {
            *top &= u8::MAX >> (8 * len - bits);
        }
        Ok(Self::Challenge::from_bytes_le_reduced(bytes))
    }

    /// Squeezes `out.len()` raw bytes.
    ///
    /// Protocol challenges come from the decoding methods above; this one
    /// backs [`challenge_short_scalar`](Self::challenge_short_scalar) and the
    /// proof-of-work check of [`check_grind`](Self::check_grind).
    fn challenge_bytes(&mut self, out: &mut [u8]);

    /// A proof of work of `bits` bits, for the prover. Returns the nonce the
    /// proof carries, or `None` for `bits == 0`, which leaves the transcript
    /// unchanged.
    ///
    /// The nonce is the least one below `2^(bits + 7)` that passes
    /// [`check_grind`](Self::check_grind), which is then applied to this
    /// transcript.
    ///
    /// # Errors
    /// [`ProofOfWorkError`], before touching the transcript, if
    /// `bits > MAX_POW_BITS` or the search exhausts the nonce space
    /// (probability at most `e^-128` for an ideal hash).
    fn grind(&mut self, bits: u32) -> Result<Option<u32>, ProofOfWorkError> {
        if bits == 0 {
            return Ok(None);
        }
        let pow = SpongePow::new(bits).ok_or(ProofOfWorkError)?;
        let nonce = pow
            .search(&TranscriptProbe(&*self))
            .ok_or(ProofOfWorkError)?;
        self.check_grind(bits, Some(nonce))?;
        Ok(Some(nonce))
    }

    /// Checks a proof of work of `bits` bits whose nonce the proof carries.
    ///
    /// For `bits == 0` the proof carries no nonce and the transcript is
    /// unchanged. For `1 <= bits <= MAX_POW_BITS` the nonce must be below
    /// `2^(bits + 7)`; it is absorbed as one [`append_bytes`] call of its
    /// canonical LEB128 encoding, then `ceil(bits / 8)` bytes are squeezed
    /// with [`challenge_bytes`] and their low `bits` bits must be zero.
    ///
    /// The step protects the challenges drawn after it, up to the next
    /// absorbed prover message. The difficulty and the position of the step
    /// must be fixed by the protocol.
    ///
    /// # Security
    ///
    /// With an ideal hash, each attempt passes with probability `2^-bits`,
    /// and the protected challenges come from squeezes after the check.
    ///
    /// # Errors
    /// [`ProofOfWorkError`] if `bits > MAX_POW_BITS`, if a nonce is missing,
    /// present at `bits == 0` or out of range, or if it fails the check. The
    /// transcript may have advanced; a rejected proof must not be continued.
    ///
    /// [`append_bytes`]: Self::append_bytes
    /// [`challenge_bytes`]: Self::challenge_bytes
    fn check_grind(&mut self, bits: u32, nonce: Option<u32>) -> Result<(), ProofOfWorkError> {
        let Some(pow) = SpongePow::new(bits) else {
            return if bits == 0 && nonce.is_none() {
                Ok(())
            } else {
                Err(ProofOfWorkError)
            };
        };
        let nonce = nonce
            .filter(|&nonce| nonce <= pow.last_nonce())
            .ok_or(ProofOfWorkError)?;
        self.append_bytes(Leb128::new(nonce).as_bytes());
        let mut word = [0u8; 4];
        let check = word.get_mut(..pow.word_len()).ok_or(ProofOfWorkError)?;
        self.challenge_bytes(check);
        if pow.accepts(check) {
            Ok(())
        } else {
            Err(ProofOfWorkError)
        }
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

/// A copy of a transcript, for the nonce search of [`Transcript::grind`]:
/// a candidate is absorbed exactly as [`Transcript::check_grind`] absorbs it.
/// Squeezes [`challenge_width`](CanonicalEncoding::challenge_width) bytes for
/// the `draw`-th challenge since the last absorb and decodes them. The bytes
/// live on the stack: the widest challenge is 68 bytes, a 254-bit field after
/// 2^64 draws.
#[cfg(any(feature = "spongefish", feature = "digest"))]
pub(crate) fn squeeze_challenge<F: CanonicalEncoding>(
    draw: u64,
    squeeze: impl FnOnce(&mut [u8]),
) -> F {
    let width = F::challenge_width(draw);
    let mut stack = [0u8; 128];
    if let Some(buf) = stack.get_mut(..width) {
        squeeze(buf);
        F::from_challenge_bytes(buf)
    } else {
        let mut buf = vec![0u8; width];
        squeeze(&mut buf);
        F::from_challenge_bytes(&buf)
    }
}

struct TranscriptProbe<'a, T>(&'a T);

impl<T: Transcript> Probe for TranscriptProbe<'_, T> {
    fn probe(&self, candidate: &[u8], out: &mut [u8]) {
        let mut transcript = self.0.clone();
        transcript.append_bytes(candidate);
        transcript.challenge_bytes(out);
    }
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
    /// Challenges drawn since the last absorb; sets the next one's width.
    draws: u64,
    _field: PhantomData<F>,
}

#[cfg(feature = "spongefish")]
impl<H, F> Clone for SpongeTranscript<H, F>
where
    H: DuplexSpongeInit<U = u8> + Clone + Default + Send + Sync + 'static,
    F: CanonicalEncoding,
{
    fn clone(&self) -> Self {
        Self {
            sponge: self.sponge.clone(),
            draws: self.draws,
            _field: PhantomData,
        }
    }
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

    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS);

    fn new(label: &'static [u8]) -> Self {
        assert!(
            label.len() <= MAX_LABEL_LEN,
            "label must be at most {MAX_LABEL_LEN} bytes",
        );
        let mut sponge = H::init(session_id().as_bytes());
        let _ = sponge.absorb(BytesMsg(label.to_vec()).encode().as_ref());
        Self {
            sponge,
            draws: 0,
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
        self.draws = 0;
    }

    fn challenge(&mut self) -> F {
        // WARNING: this decodes squeezed bytes for every sponge — including
        // Poseidon, whose byte output is not uniform — while the split-trait
        // surface (`ChallengeCodec`) admits only uniform-byte sponges. The
        // legacy facade keeps byte challenges for in-flight consumers
        // (jolt-sumcheck, jolt-openings, jolt-crypto). Once those migrate to
        // the split-trait surface this facade goes away and the
        // inconsistency with it.
        self.draws = self.draws.saturating_add(1);
        squeeze_challenge(self.draws, |buf| {
            let _ = self.sponge.squeeze(buf);
        })
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let _ = self.sponge.squeeze(out);
    }

    fn state(&self) -> [u8; 32] {
        peek_state(&self.sponge)
    }
}
