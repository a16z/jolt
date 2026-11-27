//! The operations the prover's and verifier's spongefish states share.
//!
//! [`Transcript`] absorbs public messages and checks proof-of-work nonces;
//! [`FieldTranscript`] draws field challenges. Both are implemented on
//! spongefish's [`ProverState`] and [`VerifierState`] from sealed
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

use jolt_field::Ring;
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, ProverState, Transcript as _,
    VerificationError, VerifierState,
};

use crate::challenge::message_bytes;
use crate::codec::{word_padding, AlignedBytes, BytesMsg, INLINE_BYTES, WORD_BYTES};
use crate::pow::{nonce_word, SpongePow, CHECK_BYTES};
use crate::{ChallengeCodec, ChallengeSetSize, UniformBytes};
use sealed::{Sealed, SealedField};

pub(crate) mod sealed {
    use spongefish::{Encoding, VerificationError};

    /// The sponge operations every transcript method is built from.
    pub trait Sealed {
        /// Absorbs `message` as a public message.
        fn absorb<M: Encoding + ?Sized>(&mut self, message: &M);

        /// Squeezes `len` bytes and decodes them with `decode`.
        fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T;

        /// Squeezes a fixed-size tape, preserving the wrapper's squeeze hooks.
        fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
            self.squeeze(N, |bytes| {
                let mut tape = [0; N];
                tape.copy_from_slice(bytes);
                tape
            })
        }

        /// `Ok` when `ok` holds. On failure the verifier also poisons its
        /// state, so later reads and `check_eof` fail.
        fn require(&mut self, ok: bool) -> Result<(), VerificationError>;
    }

    /// Restricts field draws to supported transcript and field pairs.
    pub trait SealedField<F> {}
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
    /// empty and the message carries the bytes that one absorb per value
    /// would. Sponges that absorb bytes without framing (Keccak, Blake2b) then
    /// reach the same state either way; the Poseidon sponge frames each absorb
    /// with its length, so for it the two differ.
    /// Shorter encodings, such as an odd count of 4-byte elements, are padded
    /// once at the end. An empty slice absorbs nothing.
    fn append_values<M: Encoding>(&mut self, values: &[M]) {
        if values.is_empty() {
            return;
        }
        let mut inline = AlignedBytes::<INLINE_BYTES>::default();
        let mut len = 0;
        let mut values = values.iter();
        while let Some(value) = values.next() {
            let encoding = value.encode();
            let encoded = encoding.as_ref();
            let end = len + encoded.len();
            if let Some(dst) = inline.0.get_mut(len..end) {
                dst.copy_from_slice(encoded);
                len = end;
            } else {
                let mut bytes = Vec::with_capacity(end + word_padding(end));
                bytes.extend_from_slice(inline.0.split_at(len).0);
                bytes.extend_from_slice(encoded);
                // The next encoding may borrow the same storage.
                drop(encoding);
                for value in values {
                    bytes.extend_from_slice(value.encode().as_ref());
                }
                bytes.resize(bytes.len() + word_padding(bytes.len()), 0);
                self.absorb(&WordAligned(&bytes));
                return;
            }
        }
        self.absorb(&WordAligned(inline.0.split_at(len + word_padding(len)).0));
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
    /// public message, then eight bytes are squeezed and the low `bits` bits
    /// of their little-endian `u64` must be zero. This is the layout of
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
        let passes = pow.accepts(&self.squeeze_array::<CHECK_BYTES>());
        self.require(passes)
    }
}

impl<T: Sealed + ?Sized> Transcript for T {}

/// Bytes already padded to a whole number of words, absorbed as they are.
struct WordAligned<'a>(&'a [u8]);

impl Encoding for WordAligned<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}

/// A transcript that draws challenges in `F`.
///
/// [`Transcript`] stays sealed, and this release also seals each
/// `FieldTranscript<F>` pair. To extend field challenge support, implement
/// the unsafe [`UniformBytes`] trait for a byte sponge or
/// [`RuntimePrimeChallenge`](crate::RuntimePrimeChallenge) for a supported
/// prime field.
///
/// ```compile_fail,E0277
/// use jolt_transcript::{ChallengeSetSize, FieldTranscript, RuntimePrimeChallenge,
///     VerificationError, VerifierState, RUNTIME_PRIME_WIDE_BYTES};
/// use spongefish::DuplexSpongeInterface;
///
/// #[derive(Clone)]
/// struct LocalSponge;
/// impl DuplexSpongeInterface for LocalSponge {
///     type U = u8;
///     fn absorb(&mut self, _: &[u8]) -> &mut Self { unimplemented!() }
///     fn squeeze(&mut self, _: &mut [u8]) -> &mut Self { unimplemented!() }
/// }
///
/// struct LocalField(bool);
/// impl RuntimePrimeChallenge for LocalField {
///     fn modulus() -> u128 { 2 }
///     fn from_le_bytes_wide(&[low, ..]: &[u8; RUNTIME_PRIME_WIDE_BYTES]) -> Self {
///         Self(low & 1 == 1)
///     }
/// }
///
/// impl FieldTranscript<LocalField> for VerifierState<'_, LocalSponge> {
///     fn challenge_set() -> ChallengeSetSize { unimplemented!() }
///     fn challenge_vector(&mut self, _: usize) -> Result<Vec<LocalField>, VerificationError> {
///         unimplemented!()
///     }
/// }
/// ```
pub trait FieldTranscript<F>: Transcript + SealedField<F> {
    /// Size of the set every challenge is drawn from: the codec's
    /// [`challenge_set_size`](ChallengeCodec::challenge_set_size). Prices the
    /// proof of work before a challenge ([`grind_bits_for_loss`]).
    ///
    /// This method replaces the `CHALLENGE_SET` associated constant.
    /// Callers must now read the size at runtime.
    ///
    /// [`grind_bits_for_loss`]: crate::grind_bits_for_loss
    fn challenge_set() -> ChallengeSetSize;

    /// Draws `len` challenges as one message of `F`'s [`ChallengeCodec`].
    ///
    /// `len == 0` returns an empty vector without touching the sponge.
    ///
    /// # Errors
    /// `VerificationError` before any sponge operation when the codec does
    /// not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes. Decoding can
    /// fail on rejection-pool exhaustion (probability at most 2^-161) or if
    /// the runtime-prime codec cannot reserve the output vector. On error the
    /// verifier's state is poisoned and `check_eof` fails too.
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
    /// Returns `VerificationError` if the powers cannot be reserved before the
    /// draw, or if [`challenge`](Self::challenge) fails. On error the verifier's
    /// state is poisoned and `check_eof` fails too.
    fn challenge_powers(&mut self, len: usize) -> Result<Vec<F>, VerificationError>
    where
        F: Ring,
    {
        let mut powers = Vec::new();
        self.require(powers.try_reserve_exact(len).is_ok())?;
        let gamma = self.challenge()?;
        powers.extend(
            std::iter::successors(Some(F::from_u64(1)), |power| Some(*power * gamma)).take(len),
        );
        Ok(powers)
    }
}

/// Squeezes one codec message of `len` challenges in `F` and decodes it.
fn draw<F, U>(
    transcript: &mut (impl Sealed + ?Sized),
    len: usize,
    decode: impl FnOnce(&[u8]) -> Option<U>,
) -> Result<U, VerificationError>
where
    F: ChallengeCodec,
{
    // A rejected length or tape poisons the verifier even if the caller
    // discards the error and continues to check_eof.
    let bytes = message_bytes::<F>(len);
    transcript.require(bytes.is_ok())?;
    // An empty message squeezes nothing. Single-challenge tapes: 56 bytes for
    // BN254, 32 for Akita's Prime128Offset6A23801F, and 8, 16 and 24 for the
    // binary fields and Prime64Offset59. Other lengths take spongefish's
    // boxed squeeze.
    let challenges = match bytes? {
        0 => decode(&[]),
        8 => decode(&transcript.squeeze_array::<8>()),
        16 => decode(&transcript.squeeze_array::<16>()),
        24 => decode(&transcript.squeeze_array::<24>()),
        32 => decode(&transcript.squeeze_array::<32>()),
        56 => decode(&transcript.squeeze_array::<56>()),
        bytes => transcript.squeeze(bytes, decode),
    };
    transcript.require(challenges.is_some())?;
    challenges.ok_or(VerificationError)
}

impl<H, R> Sealed for ProverState<H, R>
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

    fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
        self.verifier_message::<AlignedBytes<N>>().0
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        if ok {
            Ok(())
        } else {
            Err(VerificationError)
        }
    }
}

impl<H> Sealed for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        VerifierState::public_message(self, message);
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T {
        VerifierState::verifier_message_as(self, len, decode)
    }

    fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
        self.verifier_message::<AlignedBytes<N>>().0
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        self.check(|| ok)
    }
}

impl<H, R, F> SealedField<F> for ProverState<H, R>
where
    H: UniformBytes,
    R: DuplexSpongeInit<U = u8>,
    F: ChallengeCodec,
{
}

impl<H, R, F> FieldTranscript<F> for ProverState<H, R>
where
    H: UniformBytes,
    R: DuplexSpongeInit<U = u8>,
    F: ChallengeCodec,
{
    fn challenge_set() -> ChallengeSetSize {
        F::challenge_set_size()
    }

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw::<F, _>(self, len, |tape| F::decode_challenges(tape, len))
    }

    fn challenge(&mut self) -> Result<F, VerificationError> {
        draw::<F, _>(self, 1, F::decode_challenge)
    }
}

impl<H: UniformBytes, F: ChallengeCodec> SealedField<F> for VerifierState<'_, H> {}

impl<H, F> FieldTranscript<F> for VerifierState<'_, H>
where
    H: UniformBytes,
    F: ChallengeCodec,
{
    fn challenge_set() -> ChallengeSetSize {
        F::challenge_set_size()
    }

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw::<F, _>(self, len, |tape| F::decode_challenges(tape, len))
    }

    fn challenge(&mut self) -> Result<F, VerificationError> {
        draw::<F, _>(self, 1, F::decode_challenge)
    }
}

#[cfg(all(test, feature = "solinas", feature = "transcript-keccak"))]
mod tests {
    use jolt_field::Prime64Offset59;

    use super::FieldTranscript;
    use crate::challenge::test_codec::ExhaustedPrime64;
    use crate::{
        protocol_tag, prover_transcript, verifier_transcript, BytesMsg, Keccak, ProverTranscript,
    };

    const SESSION: [u8; 32] = protocol_tag(b"jolt/challenge-codecs/v1");
    const INSTANCE: [u8; 32] = [0x42; 32];

    #[test]
    fn exhaustion_fails_on_both_sides_and_poisons_the_verifier() {
        let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        ExhaustedPrime64::assert_exhausted(|| {
            FieldTranscript::<ExhaustedPrime64>::challenge_vector(&mut prover, 3)
        });
        prover.send(&BytesMsg(b"after".to_vec()));
        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, prover.narg_string());
        ExhaustedPrime64::assert_exhausted(|| {
            FieldTranscript::<ExhaustedPrime64>::challenge_vector(&mut verifier, 3)
        });
        assert!(verifier.prover_message::<BytesMsg>().is_err());
        assert!(verifier.check_eof().is_err());
    }

    #[test]
    fn scalar_exhaustion_fails_on_both_sides_and_poisons_the_verifier() {
        let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        ExhaustedPrime64::assert_exhausted(|| {
            FieldTranscript::<ExhaustedPrime64>::challenge(&mut prover)
        });
        prover.send(&BytesMsg(b"after".to_vec()));
        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, prover.narg_string());
        ExhaustedPrime64::assert_exhausted(|| {
            FieldTranscript::<ExhaustedPrime64>::challenge(&mut verifier)
        });
        assert!(verifier.prover_message::<BytesMsg>().is_err());
        assert!(verifier.check_eof().is_err());
    }

    #[test]
    fn power_capacity_overflow_poisons_the_verifier() {
        let len = isize::MAX.unsigned_abs() / size_of::<Prime64Offset59>() + 1;
        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, &[]);
        let powers: Result<Vec<Prime64Offset59>, _> = verifier.challenge_powers(len);
        assert!(powers.is_err());
        assert!(verifier.check_eof().is_err());
    }
}
