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

use jolt_field::{CanonicalEncoding, Ring};
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, ProverState, Transcript as _,
    VerificationError, VerifierState,
};

use crate::challenge::message_bytes;
use crate::codec::{word_padding, BytesMsg, WORD_BYTES};
use crate::pow::{nonce_word, SpongePow, CHECK_BYTES};
use crate::{
    challenge_bits_for_loss, ChallengeCodec, ChallengeSetSize, UniformBytes, SECURITY_BITS,
};
use sealed::Sealed;

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
    /// empty and the sponge absorbs the bytes that one absorb per value would.
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
        nonce: Option<u32>,
    ) -> Result<(), VerificationError> {
        let Some(pow) = SpongePow::new(bits) else {
            return self.require(bits == 0 && nonce.is_none());
        };
        let Some(nonce) = nonce.filter(|&nonce| nonce <= pow.last_nonce()) else {
            return self.require(false);
        };
        self.absorb(&nonce_word(nonce));
        let passes = self.squeeze(CHECK_BYTES, |check| pow.accepts(check));
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

    /// Draws one challenge on the integers `[0, 2^b)`, for a step whose bad
    /// set holds at most `loss_factor` challenges: `b` is
    /// [`challenge_bits_for_loss`] at [`SECURITY_BITS`], the narrowest width
    /// that needs no proof of work. A group scalar this short costs about
    /// half as much as a full-width BN254 element.
    ///
    /// It squeezes `ceil(b / 8)` bytes rounded up to a whole word, keeps the
    /// low `b` bits of their little-endian integer, and decodes that. The
    /// value depends only on those `b` bits, so the extra bytes change
    /// nothing but the squeeze's alignment. As `b` is below the bit length of
    /// the field order, distinct integers decode to distinct elements. The
    /// sponge's bytes are uniform ([`UniformBytes`]), so the integer is too.
    ///
    /// # Errors
    /// `VerificationError`, before any sponge operation, when
    /// [`challenge_bits_for_loss`] rejects `loss_factor` or `b` is not below
    /// the bit length of the field order. The verifier's state is poisoned
    /// then, so `check_eof` fails too.
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

/// One codec message of `len` challenges: the single implementation behind
/// every [`FieldTranscript`].
pub(crate) fn draw_challenges<T, F>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<F>, VerificationError>
where
    T: Sealed + ?Sized,
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
    Ok(transcript.squeeze(short_scalar_squeeze(bits), |tape| {
        F::from_bytes_le_reduced(&short_scalar_bytes(tape, bits))
    }))
}

/// Bytes squeezed for a `bits`-bit short scalar: `ceil(bits / 8)` rounded up
/// to a whole word.
pub(crate) const fn short_scalar_squeeze(bits: u32) -> usize {
    (bits as usize).div_ceil(8).div_ceil(WORD_BYTES) * WORD_BYTES
}

/// The low `bits` bits of `tape`'s little-endian integer, as 32 LE bytes.
/// Every byte at index `ceil(bits / 8)` or later is zero, and the top partial
/// byte keeps only its low `bits mod 8` bits, so the result depends only on
/// the low `bits` bits whatever the tape's length. `bits <= 256`.
pub(crate) fn short_scalar_bytes(tape: &[u8], bits: u32) -> [u8; 32] {
    let len = (bits as usize).div_ceil(8);
    let mut word = [0u8; 32];
    for (out, byte) in word.iter_mut().zip(tape).take(len) {
        *out = *byte;
    }
    // `8 * len - bits` is in `0..8`, so the shift cannot overflow.
    let excess = (8 * len as u32).saturating_sub(bits) % 8;
    if let Some(top) = len.checked_sub(1).and_then(|top| word.get_mut(top)) {
        *top &= u8::MAX >> excess;
    }
    word
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

    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<F, VerificationError>
    where
        F: CanonicalEncoding,
    {
        draw_short_scalar(self, loss_factor)
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

    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<F, VerificationError>
    where
        F: CanonicalEncoding,
    {
        draw_short_scalar(self, loss_factor)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{short_scalar_bytes, short_scalar_squeeze};
    use crate::{challenge_bits_for_loss, SECURITY_BITS};

    /// The little-endian integer of `bytes` is below `2^bits`.
    fn below_power_of_two(bytes: &[u8; 32], bits: u32) -> bool {
        let full = bits as usize / 8;
        let partial = bits % 8;
        bytes
            .iter()
            .enumerate()
            .all(|(i, &byte)| match i.cmp(&full) {
                core::cmp::Ordering::Less => true,
                core::cmp::Ordering::Equal => u32::from(byte) < 1 << partial,
                core::cmp::Ordering::Greater => byte == 0,
            })
    }

    /// Every short width squeezes whole words, and an all-ones tape of that
    /// length decodes below `2^b`: the extra bytes are cleared, not just the
    /// top byte masked.
    #[test]
    fn short_scalars_squeeze_words_and_keep_only_the_low_bits() {
        let mut widths: Vec<u32> = (1..=255).collect();
        // The Dory widths: losses 2, 3 and 4.
        for loss in [2, 3, 4] {
            let bits = challenge_bits_for_loss(loss, SECURITY_BITS).unwrap();
            assert!(bits == 129 || bits == 130, "loss {loss}: {bits} bits");
            assert_eq!(short_scalar_squeeze(bits), 24);
            widths.push(bits);
        }
        for bits in widths {
            let len = short_scalar_squeeze(bits);
            assert_eq!(len % 8, 0, "{bits} bits");
            assert!(len >= (bits as usize).div_ceil(8) && len <= 32);
            let ones = vec![0xffu8; len];
            let value = short_scalar_bytes(&ones, bits);
            assert!(below_power_of_two(&value, bits), "{bits} bits");
            // The low `bits` bits all survive.
            let set: u32 = value.iter().map(|byte| byte.count_ones()).sum();
            assert_eq!(set, bits, "{bits} bits");
            // Bytes past `ceil(bits / 8)` do not change the value.
            let mut other = ones.clone();
            for byte in other.iter_mut().skip((bits as usize).div_ceil(8)) {
                *byte = 0;
            }
            assert_eq!(short_scalar_bytes(&other, bits), value, "{bits} bits");
        }
    }
}
