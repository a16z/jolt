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
//! The transcript is positional: a message's place in the protocol's
//! schedule says what it is, so no message carries a label. The sponge
//! absorbs a message's bytes with no length or separator, so the messages
//! absorbed between two squeezes must decode uniquely from their
//! concatenation, given everything the verifier knows before it reads them.
//! Every [`Encoding`] a protocol absorbs is therefore of fixed width, or of a
//! length fixed by bound data before the proof is accepted, or carries its own
//! `u64` count or length, as [`BytesMsg`] does.
//!
//! Every absorb and every squeeze is a whole number of eight-byte words, so
//! each starts and ends on a word boundary of the sponge's rate. Encodings
//! shorter than a word are zero-padded to one; the padding is injective
//! because its length follows from the message's own length.
//!
//! That is the layout on a byte sponge. `PoseidonSponge` absorbs the same
//! byte encodings, each packed on its own with its length into BN254 `Fr`
//! elements, so it frames every message itself ([`SpongeUnit`]). There the
//! split into messages is part of the transcript: the prover and the verifier
//! must absorb the same messages, not only the same bytes.

use jolt_field::Ring;
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, FromNarg, NargReader, ProverState, Unit,
    VerificationError, VerifierState,
};

use crate::codec::{word_padding, BytesMsg, SpongeUnit, Units, WORD_BYTES};
use crate::pow::{nonce_word, SpongePow};
use crate::{ChallengeSetSize, ChallengeSponge};
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
    ///
    /// The encoding must be a whole number of eight-byte words, so the
    /// absorb ends on a word boundary. Every production field element, group
    /// element and `u64` qualifies. A value with a shorter encoding, such as a
    /// 32-bit field element, goes through [`append_values`](Self::append_values),
    /// which zero-pads it to a word.
    ///
    /// # Panics
    /// In debug builds, if the encoding's length is not a multiple of eight.
    fn append<M: Encoding + ?Sized>(&mut self, message: &M) {
        debug_assert_eq!(
            message.encode().as_ref().len() % WORD_BYTES,
            0,
            "Transcript::append needs a whole number of words; pad with append_values"
        );
        self.absorb(message);
    }

    /// Absorbs the values' encodings, concatenated in order and zero-padded
    /// to the next word, as one message with no count: the caller's schedule
    /// must fix `values.len()` from bound data. A protocol whose count the
    /// prover chooses absorbs that count as a `u64` word first.
    ///
    /// For values whose encodings are whole words, such as every production
    /// field element and Dory's compressed group elements, the padding is
    /// empty and a byte sponge absorbs the bytes that one absorb per value
    /// would. `PoseidonSponge` frames the whole slice as one message, which
    /// differs from one message per value.
    /// Shorter encodings, such as an odd count of 4-byte elements, are padded
    /// once at the end. An empty slice absorbs nothing.
    fn append_values<M: Encoding>(&mut self, values: &[M]) {
        if values.is_empty() {
            return;
        }
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(value.encode().as_ref());
        }
        bytes.resize(bytes.len() + word_padding(bytes.len()), 0);
        self.absorb(&WordAligned(bytes));
    }

    /// Absorbs `bytes` as a [`BytesMsg`]: a `u64` length, the bytes, and zero
    /// padding to the next word.
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.absorb(&BytesMsg(bytes));
    }

    /// Checks a proof of work of `bits` bits whose nonce the proof carries
    /// outside the NARG string.
    ///
    /// For `bits == 0` the proof carries no nonce and the transcript is
    /// unchanged. For `1 <= bits <= MAX_POW_BITS` the nonce must be at most
    /// `2^(bits + 7) - 1`; it is absorbed as one `u64` little-endian word in a
    /// public message, then the check is squeezed and its low `bits` bits must
    /// be zero: eight bytes read as a little-endian `u64` on a byte sponge,
    /// one element's canonical integer on `PoseidonSponge`. This is the layout of
    /// [`VerifierTranscript::check_grind`](crate::VerifierTranscript::check_grind)
    /// with the nonce absorbed as a public message instead of read from the
    /// NARG string; the sponge absorbs the same word.
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
        nonce: Option<u64>,
    ) -> Result<(), VerificationError> {
        let Some(pow) = SpongePow::new(bits) else {
            return self.require(bits == 0 && nonce.is_none());
        };
        let Some(nonce) = nonce.filter(|&nonce| nonce <= pow.last_nonce()) else {
            return self.require(false);
        };
        self.absorb(&nonce_word(nonce));
        let passes = self.squeeze(Self::Unit::CHECK_LEN, |check| pow.accepts(check));
        self.require(passes)
    }
}

impl<T: Sealed + ?Sized> Transcript for T {}

/// Bytes already padded to a whole number of words, absorbed as they are.
struct WordAligned(Vec<u8>);

impl Encoding for WordAligned {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.as_slice()
    }
}

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
}

impl<H, F> FieldTranscript<F> for VerifierState<'_, H>
where
    H: ChallengeSponge<F>,
{
    const CHALLENGE_SET: ChallengeSetSize = H::CHALLENGE_SET;

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges::<_, H, F>(self, len)
    }
}
