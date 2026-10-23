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
//!
//! A byte sponge absorbs a message's bytes with no length or separator, so
//! the messages absorbed between two squeezes must decode uniquely from their
//! concatenation. Every [`Encoding`] a protocol absorbs is therefore of fixed
//! width, or of a length fixed by bound data, or carries its own count, as
//! [`append_values`](Transcript::append_values) and [`BytesMsg`] do.
//! `PoseidonSponge` frames each message itself ([`SpongeUnit`]).

use jolt_field::{CanonicalEncoding, Ring};
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, FromNarg, NargReader, ProverState, Unit,
    VerificationError, VerifierState,
};

use crate::codec::{BytesMsg, Label, LabelWithCount, SpongeUnit, Units};
use crate::pow::{Leb128, SpongePow};
use crate::{challenge_bits_for_loss, ChallengeSetSize, ChallengeSponge, SECURITY_BITS};
use sealed::Sealed;

pub(crate) mod sealed {
    use spongefish::{Encoding, VerificationError};

    use crate::SpongeUnit;

    /// The three sponge operations every transcript method is built from.
    pub trait Sealed {
        /// The sponge's alphabet.
        type Unit: SpongeUnit;

        /// Absorbs `message` as a public message.
        fn absorb<M: Encoding + ?Sized>(&mut self, message: &M);

        /// Squeezes `len` units and decodes them with `decode`.
        fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[Self::Unit]) -> T) -> T;

        /// `Ok` when `ok` holds. On failure the verifier also poisons its
        /// state, so later reads and `check_eof` fail.
        fn require(&mut self, ok: bool) -> Result<(), VerificationError>;
    }
}

/// A Fiat-Shamir transcript: spongefish's prover or verifier state.
///
/// Every method is provided once and runs the same sponge operations on both
/// sides.
pub trait Transcript: Sealed {
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
    /// per public message, then the check is squeezed and its low `bits` bits
    /// must be zero: `ceil(bits / 8)` little-endian bytes on a byte sponge, one
    /// element's canonical integer on `PoseidonSponge`. This is the layout of
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
        let passes = self.squeeze(Self::Unit::check_len(bits), |check| pow.accepts(check));
        self.require(passes)
    }
}

impl<T: Sealed + ?Sized> Transcript for T {}

/// A transcript that draws challenges in `F`.
pub trait FieldTranscript<F>: Transcript {
    /// Size of the set every challenge is drawn from
    /// ([`ChallengeSponge::CHALLENGE_SET`]). Prices the proof of work before a
    /// challenge ([`grind_bits_for_loss`]).
    ///
    /// [`grind_bits_for_loss`]: crate::grind_bits_for_loss
    const CHALLENGE_SET: ChallengeSetSize;

    /// Draws `len` challenges as one message of the sponge's
    /// [`ChallengeSponge`] map: `F`'s [`ChallengeCodec`] on a byte sponge,
    /// squeezed elements on `PoseidonSponge`.
    ///
    /// `len == 0` returns an empty vector without touching the sponge.
    ///
    /// # Errors
    /// `VerificationError`, before any sponge operation, when the map does
    /// not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes; otherwise on
    /// rejection-pool exhaustion (probability at most 2^-161). The verifier's
    /// state is poisoned then, so `check_eof` fails too.
    ///
    /// [`ChallengeCodec`]: crate::ChallengeCodec
    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError>;

    /// Draws one challenge, as a message of length one.
    ///
    /// # Errors
    /// As [`challenge_vector`](Self::challenge_vector).
    fn challenge(&mut self) -> Result<F, VerificationError> {
        self.challenge_vector(1)?.pop().ok_or(VerificationError)
    }

    /// Draws one challenge on the integers `[0, 2^b)`, for a step whose bad
    /// set holds at most `loss_factor` challenges: `b` is
    /// [`challenge_bits_for_loss`] at [`SECURITY_BITS`], the narrowest width
    /// that needs no proof of work. A group scalar this short costs about
    /// half as much as a full-width BN254 element.
    ///
    /// It reads an integer from the sponge ([`SpongeUnit::short_word`]),
    /// clears the bits above `b`, and decodes the little-endian integer. A
    /// byte sponge squeezes `ceil(b / 8)` bytes, which are uniform
    /// ([`UniformBytes`]). `PoseidonSponge` squeezes two elements `x0` and
    /// `x1` and keeps the low `b` bits of `x0 + p * x1`, within 2^-253 of
    /// uniform. As `b` is below the bit length of the field order, distinct
    /// integers decode to distinct elements.
    ///
    /// # Errors
    /// `VerificationError`, before any sponge operation, when
    /// [`challenge_bits_for_loss`] rejects `loss_factor` or `b` is not below
    /// the bit length of the field order. The verifier's state is poisoned
    /// then, so `check_eof` fails too.
    ///
    /// [`UniformBytes`]: crate::UniformBytes
    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<F, VerificationError>
    where
        F: CanonicalEncoding;

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

/// One message of `len` challenges from sponge `H`: the single
/// implementation behind every [`FieldTranscript`].
fn draw_challenges<T, H, F>(transcript: &mut T, len: usize) -> Result<Vec<F>, VerificationError>
where
    T: Sealed<Unit = H::U> + ?Sized,
    H: ChallengeSponge<F>,
{
    // A length the map rejects poisons the verifier too, so a caller that
    // drops the error still fails `check_eof`.
    let units = H::challenge_units(len);
    transcript.require(units.is_some())?;
    let units = units.ok_or(VerificationError)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let challenges = transcript.squeeze(units, |tape| H::decode_challenges(tape, len));
    transcript.require(challenges.is_some())?;
    challenges.ok_or(VerificationError)
}

/// One short challenge of [`challenge_short_scalar`]: the single
/// implementation behind every [`FieldTranscript`].
///
/// [`challenge_short_scalar`]: FieldTranscript::challenge_short_scalar
pub(crate) fn draw_short_scalar<T, F>(
    transcript: &mut T,
    loss_factor: u64,
) -> Result<F, VerificationError>
where
    T: Sealed + ?Sized,
    F: CanonicalEncoding,
{
    // A width that does not fit poisons the verifier too, so a caller that
    // drops the error still fails `check_eof`.
    let bits = challenge_bits_for_loss(loss_factor, SECURITY_BITS)
        .ok()
        .filter(|&bits| bits < F::MODULUS_BITS);
    transcript.require(bits.is_some())?;
    let bits = bits.ok_or(VerificationError)?;
    // `bits < MODULUS_BITS <= 256`, so the bytes fit one 32-byte word.
    let len = bits.div_ceil(8) as usize;
    let excess = 8 * len as u32 - bits;
    let scalar = transcript.squeeze(T::Unit::short_len(bits), |tape| {
        let mut word = T::Unit::short_word(tape)?;
        let bytes = word.get_mut(..len)?;
        if let Some(top) = bytes.last_mut() {
            *top &= u8::MAX >> excess;
        }
        Some(F::from_bytes_le_reduced(bytes))
    });
    transcript.require(scalar.is_some())?;
    scalar.ok_or(VerificationError)
}

impl<H, R> Sealed for ProverState<H, R>
where
    H: DuplexSpongeInterface<U: SpongeUnit>,
    R: DuplexSpongeInit<U = u8>,
{
    type Unit = H::U;

    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        ProverState::public_message(self, &Units(message));
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[H::U]) -> T) -> T {
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

/// A prover message that never parses and absorbs nothing: reading it is how
/// [`Sealed::require`] poisons a verifier over any alphabet. Spongefish's
/// `Transcript::check` does the same, but only over bytes.
struct Rejection;

impl<U: Unit> Encoding<U> for Rejection {
    fn encode(&self) -> impl AsRef<[U]> {
        let empty: [U; 0] = [];
        empty
    }
}

impl FromNarg for Rejection {
    fn from_narg(_: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        Err(VerificationError)
    }
}

impl<H> Sealed for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U: SpongeUnit>,
{
    type Unit = H::U;

    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        VerifierState::public_message(self, &Units(message));
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[H::U]) -> T) -> T {
        VerifierState::verifier_message_as(self, len, decode)
    }

    /// Neither branch absorbs or squeezes. Reading no messages fails only on a
    /// poisoned state; reading a [`Rejection`] always fails and poisons it.
    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        if ok {
            self.prover_messages_vec::<Rejection>(0).map(drop)
        } else {
            self.prover_message::<Rejection>().map(drop)
        }
    }
}

impl<H, R, F> FieldTranscript<F> for ProverState<H, R>
where
    H: ChallengeSponge<F>,
    R: DuplexSpongeInit<U = u8>,
{
    const CHALLENGE_SET: ChallengeSetSize = H::CHALLENGE_SET;

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges::<_, H, F>(self, len)
    }

    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<F, VerificationError>
    where
        F: CanonicalEncoding,
    {
        draw_short_scalar(self, loss_factor)
    }
}

impl<H, F> FieldTranscript<F> for VerifierState<'_, H>
where
    H: ChallengeSponge<F>,
{
    const CHALLENGE_SET: ChallengeSetSize = H::CHALLENGE_SET;

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges::<_, H, F>(self, len)
    }

    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<F, VerificationError>
    where
        F: CanonicalEncoding,
    {
        draw_short_scalar(self, loss_factor)
    }
}
