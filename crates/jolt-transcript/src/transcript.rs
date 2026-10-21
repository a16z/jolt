//! The operations the prover's and verifier's spongefish states share.
//!
//! [`Transcript`] absorbs public messages and checks proof-of-work nonces;
//! [`FieldTranscript`] draws field challenges. Both are implemented on
//! spongefish's [`ProverState`] and [`VerifierState`] from three sealed
//! sponge operations, so every method has one implementation that the two
//! sides run alike. The traits are sealed: a transcript that drew challenges
//! some other way would void the codecs' bounds.
//!
//! A public message is absorbed by both sides and writes no proof bytes. A
//! protocol whose proof is a serialized structure absorbs each field it reads
//! from that structure as a public message; spongefish absorbs the same bytes
//! for a public message and for a prover message, so the challenges are those
//! of the same protocol with the field sent through the NARG string.

use jolt_field::Ring;
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, ProverState, Transcript as _,
    VerificationError, VerifierState,
};

use crate::challenge::message_bytes;
use crate::codec::{BytesMsg, Label, LabelWithCount};
use crate::pow::{Leb128, SpongePow};
use crate::{ChallengeCodec, ChallengeSetSize, UniformBytes};

pub(crate) mod sealed {
    use spongefish::{Encoding, VerificationError};

    /// The three sponge operations every transcript method is built from.
    pub trait Sealed {
        /// Absorbs `message` as a public message.
        fn absorb<M: Encoding + ?Sized>(&mut self, message: &M);

        /// Squeezes `len` bytes and decodes them with `decode`.
        fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T;

        /// `Ok` when `ok` holds. On failure the verifier also poisons its
        /// state, so later reads and `check_eof` fail.
        fn require(&mut self, ok: bool) -> Result<(), VerificationError>;
    }
}

/// A Fiat-Shamir transcript: spongefish's prover or verifier state.
///
/// Every method is provided once and runs the same sponge operations on both
/// sides.
pub trait Transcript: sealed::Sealed {
    /// Absorbs `message` as a public message.
    fn append<M: Encoding + ?Sized>(&mut self, message: &M) {
        self.absorb(message);
    }

    /// Absorbs a [`Label`] word, then `message`.
    fn append_labeled<M: Encoding + ?Sized>(&mut self, label: &'static [u8], message: &M) {
        self.absorb(&Label(label));
        self.absorb(message);
    }

    /// Absorbs a [`LabelWithCount`] word carrying `values.len()`, then each
    /// value in order.
    fn append_values<M: Encoding>(&mut self, label: &'static [u8], values: &[M]) {
        self.absorb(&LabelWithCount(label, values.len() as u64));
        for value in values {
            self.absorb(value);
        }
    }

    /// Absorbs `bytes` as a length-prefixed [`BytesMsg`].
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.absorb(&BytesMsg(bytes));
    }

    /// Checks a proof of work of `bits` bits whose nonce the proof carries
    /// outside the NARG string.
    ///
    /// For `bits == 0` the proof carries no nonce and the transcript is
    /// unchanged. For `1 <= bits <= MAX_POW_BITS` the nonce must be at most
    /// `2^(bits + 7) - 1`; its canonical LEB128 encoding is absorbed one byte
    /// per public message, then `ceil(bits / 8)` bytes are squeezed and their
    /// low `bits` bits must be zero. This is the layout of
    /// [`VerifierTranscript::check_grind`](crate::VerifierTranscript::check_grind)
    /// with the nonce absorbed as a public message instead of read from the
    /// NARG string; the sponge absorbs the same bytes.
    ///
    /// The step protects the challenges drawn after it, up to the next
    /// absorbed message. The difficulty and the position of the step must be
    /// fixed by the protocol.
    ///
    /// # Errors
    /// `VerificationError` if `bits > MAX_POW_BITS`, if a nonce is missing,
    /// present at `bits == 0` or out of range, or if it fails the check. The
    /// verifier's state is poisoned then, so `check_eof` fails too.
    fn check_public_grind(
        &mut self,
        bits: u32,
        nonce: Option<u32>,
    ) -> Result<(), VerificationError> {
        let Some(pow) = SpongePow::new(bits) else {
            return self.require(bits == 0 && nonce.is_none());
        };
        let Some(nonce) = nonce.filter(|&nonce| nonce <= pow.last_nonce()) else {
            return self.require(false);
        };
        for byte in Leb128::new(nonce).as_bytes() {
            self.absorb(byte);
        }
        let passes = self.squeeze(pow.word_len(), |check| pow.accepts(check));
        self.require(passes)
    }
}

impl<T: sealed::Sealed + ?Sized> Transcript for T {}

/// A transcript that draws challenges in `F`.
pub trait FieldTranscript<F>: Transcript {
    /// Size of the set every challenge is drawn from: the codec's
    /// [`CHALLENGE_SET_SIZE`](ChallengeCodec::CHALLENGE_SET_SIZE). Prices the
    /// proof of work before a challenge ([`grind_bits_for_loss`]).
    ///
    /// [`grind_bits_for_loss`]: crate::grind_bits_for_loss
    const CHALLENGE_SET: ChallengeSetSize;

    /// Draws `len` challenges as one message of `F`'s [`ChallengeCodec`].
    ///
    /// `len == 0` returns an empty vector without touching the sponge.
    ///
    /// # Errors
    /// `VerificationError`, before any sponge operation, when the codec does
    /// not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes; otherwise on
    /// rejection-pool exhaustion (probability at most 2^-161). The verifier's
    /// state is poisoned then, so `check_eof` fails too.
    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError>;

    /// Draws one challenge, as a message of length one.
    ///
    /// # Errors
    /// As [`challenge_vector`](Self::challenge_vector).
    fn challenge(&mut self) -> Result<F, VerificationError> {
        self.challenge_vector(1)?.pop().ok_or(VerificationError)
    }

    /// Draws one challenge `gamma` and returns `[1, gamma, ..., gamma^(len-1)]`.
    /// `gamma` is drawn even for `len == 0`.
    ///
    /// # Errors
    /// As [`challenge_vector`](Self::challenge_vector).
    fn challenge_powers(&mut self, len: usize) -> Result<Vec<F>, VerificationError>
    where
        F: Ring,
    {
        let gamma = self.challenge()?;
        Ok(
            std::iter::successors(Some(F::from_u64(1)), |power| Some(*power * gamma))
                .take(len)
                .collect(),
        )
    }
}

/// One codec message of `len` challenges: the single implementation behind
/// every [`FieldTranscript`].
pub(crate) fn draw_challenges<T, F>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<F>, VerificationError>
where
    T: sealed::Sealed + ?Sized,
    F: ChallengeCodec,
{
    // A length the codec rejects poisons the verifier too, so a caller that
    // drops the error still fails `check_eof`.
    let bytes = message_bytes::<F>(len);
    transcript.require(bytes.is_ok())?;
    let bytes = bytes?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let challenges = transcript.squeeze(bytes, |tape| F::decode_challenges(tape, len));
    transcript.require(challenges.is_some())?;
    challenges.ok_or(VerificationError)
}

impl<H, R> sealed::Sealed for ProverState<H, R>
where
    H: DuplexSpongeInterface<U = u8>,
    R: DuplexSpongeInit<U = u8>,
{
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        ProverState::public_message(self, message);
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T {
        ProverState::verifier_message_as(self, len, decode)
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        if ok {
            Ok(())
        } else {
            Err(VerificationError)
        }
    }
}

impl<H> sealed::Sealed for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        VerifierState::public_message(self, message);
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T {
        VerifierState::verifier_message_as(self, len, decode)
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        self.check(|| ok)
    }
}

impl<H, R, F> FieldTranscript<F> for ProverState<H, R>
where
    H: UniformBytes,
    R: DuplexSpongeInit<U = u8>,
    F: ChallengeCodec,
{
    const CHALLENGE_SET: ChallengeSetSize = F::CHALLENGE_SET_SIZE;

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges(self, len)
    }
}

impl<H, F> FieldTranscript<F> for VerifierState<'_, H>
where
    H: UniformBytes,
    F: ChallengeCodec,
{
    const CHALLENGE_SET: ChallengeSetSize = F::CHALLENGE_SET_SIZE;

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges(self, len)
    }
}
