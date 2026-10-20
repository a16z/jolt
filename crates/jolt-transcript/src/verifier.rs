//! Spongefish-native [`VerifierTranscript`] surface.

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use crate::Blake2b512;
use crate::{challenge::message_bytes, pow::SpongePow, ChallengeCodec, UniformBytes};
#[cfg(all(
    feature = "bn254",
    any(feature = "transcript-blake2b", feature = "transcript-keccak")
))]
use jolt_field::Fr;
#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
use spongefish::instantiations::Keccak;
use spongefish::{
    DuplexSpongeInterface, Encoding, FromNarg, FromUniform, Transcript, VerificationError,
    VerifierState,
};

#[cfg(all(
    feature = "bn254",
    any(feature = "transcript-blake2b", feature = "transcript-keccak")
))]
use crate::prover::OptimizedChallenge;

/// Verifier-side spongefish transcript.
pub trait VerifierTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the prover.
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T);

    /// Reads a prover message from the NARG, absorbing it into the sponge.
    fn prover_message<T: Encoding<H::U> + FromNarg>(&mut self) -> Result<T, VerificationError>;

    /// Squeezes a verifier challenge.
    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T;

    /// Draws one message with conventional TV ≤ 2^-161 across all `len` elements.
    /// The protocol session/version must bind the codec, field, and length.
    /// `len == 0` returns an empty vector without touching the sponge. Only
    /// sponges with uniform output qualify ([`UniformBytes`]); `PoseidonSponge`
    /// does not.
    ///
    /// # Errors
    /// Returns `VerificationError`, before any sponge operation, when the codec
    /// does not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes. Otherwise on
    /// rejection-pool exhaustion (probability at most 2^-161). Every failure
    /// poisons the state, so later reads and `check_eof` fail.
    fn field_challenges<F: ChallengeCodec>(
        &mut self,
        len: usize,
    ) -> Result<Vec<F>, VerificationError>
    where
        H: UniformBytes;

    /// Checks a proof of work of `bits` bits on this transcript's own sponge.
    ///
    /// For `bits == 0` this does nothing. For `1 <= bits <= MAX_POW_BITS`, it
    /// reads a nonce below `2^(bits + 7)` as canonical LEB128, one byte per
    /// prover message, then squeezes `ceil(bits / 8)` bytes and requires their
    /// low `bits` bits to be zero. The layout is the one proposed in
    /// arkworks-rs/spongefish#301.
    ///
    /// The step protects the verifier messages drawn after it, up to the next
    /// prover message. The protocol must draw the challenges it protects
    /// directly after this step: a prover message in between would let the
    /// prover change them without redoing the work.
    ///
    /// # Security
    ///
    /// With an ideal sponge, each attempt passes the check with probability
    /// `2^-bits`, and the protected challenges come from the bytes after the
    /// check. For the Keccak duplex sponge, the reduction of Chiesa and Orrù
    /// prices this per permutation call; the Blake2b hash bridge and Poseidon
    /// need their own argument. The difficulty, the position of this step and the
    /// challenges it protects must be fixed by the protocol and bound in its
    /// session.
    ///
    /// The verifier accepts any nonce in range that passes, not only the least
    /// one, so proof bytes are not unique; an honest prover sends the least.
    ///
    /// # Errors
    /// Returns `VerificationError`, before any sponge operation, if
    /// `bits > MAX_POW_BITS`. Otherwise on a truncated, non-canonical or
    /// out-of-range nonce, or one whose check bytes are not zero. Every
    /// failure poisons the state, so later reads and `check_eof` fail.
    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H: DuplexSpongeInterface<U = u8>;

    /// Asserts the NARG was fully consumed.
    ///
    /// Soundness-critical: without this check, `valid_proof || garbage`
    /// verifies as if the trailing bytes were absent, making the top-level
    /// proof bytes malleable. Call once at the end of the top-level verify
    /// function on the success path; error paths skip it (the proof is
    /// already rejected). Composite verifiers follow the same rule at
    /// their own finalize boundary.
    ///
    /// Takes `self` by value to prevent reuse.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the NARG has unread bytes.
    fn check_eof(self) -> Result<(), VerificationError>;
}

impl<H> VerifierTranscript<H> for VerifierState<'_, H>
where
    H: DuplexSpongeInterface,
{
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T) {
        VerifierState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<H::U> + FromNarg>(&mut self) -> Result<T, VerificationError> {
        VerifierState::prover_message::<T>(self)
    }

    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T {
        VerifierState::verifier_message::<T>(self)
    }

    fn field_challenges<F: ChallengeCodec>(
        &mut self,
        len: usize,
    ) -> Result<Vec<F>, VerificationError>
    where
        H: UniformBytes,
    {
        // A length the codec rejects poisons the state too, so a caller that
        // drops the error still fails `check_eof`.
        let n = message_bytes::<F>(len);
        Transcript::check(self, || n.is_ok())?;
        let n = n?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let challenges =
            VerifierState::verifier_message_as(self, n, |bytes| F::decode_challenges(bytes, len));
        // Exhaustion poisons the state, so later reads and `check_eof` fail too.
        Transcript::check(self, || challenges.is_some())?;
        challenges.ok_or(VerificationError)
    }

    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H: DuplexSpongeInterface<U = u8>,
    {
        if bits == 0 {
            return Ok(());
        }
        let Some(pow) = SpongePow::new(bits) else {
            return Transcript::check(self, || false);
        };
        // Failed reads poison the state themselves; failed checks poison it
        // through `Transcript::check`, so `check_eof` fails afterwards too.
        let nonce = pow.read_nonce(|| VerifierState::prover_message::<u8>(self))?;
        Transcript::check(self, || nonce.is_some())?;
        let passes =
            VerifierState::verifier_message_as(self, pow.word_len(), |check| pow.accepts(check));
        Transcript::check(self, || passes)
    }

    fn check_eof(self) -> Result<(), VerificationError> {
        VerifierState::check_eof(self)
    }
}

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
impl OptimizedChallenge for VerifierState<'_, Blake2b512> {
    fn challenge_128(&mut self) -> Fr {
        Fr::from(VerifierState::verifier_message::<u128>(self))
    }
}

#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
impl OptimizedChallenge for VerifierState<'_, Keccak> {
    fn challenge_128(&mut self) -> Fr {
        Fr::from(VerifierState::verifier_message::<u128>(self))
    }
}
