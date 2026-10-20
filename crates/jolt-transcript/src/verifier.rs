//! Spongefish-native [`VerifierTranscript`] surface.

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use crate::Blake2b512;
use crate::{challenge::decode_message, ChallengeCodec};
#[cfg(all(
    feature = "bn254",
    any(feature = "transcript-blake2b", feature = "transcript-keccak")
))]
use jolt_field::Fr;
#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
use spongefish::instantiations::Keccak;
use spongefish::{
    DuplexSpongeInterface, Encoding, FromNarg, FromUniform, VerificationError, VerifierState,
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
    ///
    /// # Errors
    /// Returns `VerificationError` on length overflow, an unsupported length,
    /// or rejection-pool exhaustion (probability at most 2^-161).
    fn field_challenges<F: ChallengeCodec>(
        &mut self,
        len: usize,
    ) -> Result<Vec<F>, VerificationError>
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
        H: DuplexSpongeInterface<U = u8>,
    {
        let n = F::challenge_bytes(len).ok_or(VerificationError)?;
        VerifierState::verifier_message_as(self, n, |bytes| decode_message::<F>(bytes, len))
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
