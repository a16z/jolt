//! Spongefish-native [`VerifierTranscript`] surface.

use spongefish::{
    Decoding, DuplexSpongeInterface, Encoding, NargDeserialize, VerificationResult, VerifierState,
};

use crate::prover::OptimizedChallenge;

/// Verifier-side spongefish transcript.
pub trait VerifierTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the prover.
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T);

    /// Reads a prover message from the NARG, absorbing it into the sponge.
    fn prover_message<T: Encoding<[H::U]> + NargDeserialize>(&mut self) -> VerificationResult<T>;

    /// Squeezes a verifier challenge.
    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T;

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
    fn check_eof(self) -> VerificationResult<()>;
}

impl<H> VerifierTranscript<H> for VerifierState<'_, H>
where
    H: DuplexSpongeInterface,
{
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T) {
        VerifierState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<[H::U]> + NargDeserialize>(&mut self) -> VerificationResult<T> {
        VerifierState::prover_message::<T>(self)
    }

    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T {
        VerifierState::verifier_message::<T>(self)
    }

    fn check_eof(self) -> VerificationResult<()> {
        VerifierState::check_eof(self)
    }
}

#[cfg(feature = "transcript-blake2b")]
impl OptimizedChallenge for VerifierState<'_, spongefish::instantiations::Blake2b512> {
    fn challenge_u128(&mut self) -> u128 {
        VerifierState::verifier_message::<u128>(self)
    }
}

#[cfg(feature = "transcript-keccak")]
impl OptimizedChallenge for VerifierState<'_, spongefish::instantiations::Keccak> {
    fn challenge_u128(&mut self) -> u128 {
        VerifierState::verifier_message::<u128>(self)
    }
}

// Poseidon `OptimizedChallenge` (verifier side) — LIVE, required; mirror of the
// prover impl. D5b/DEV-43 has landed, so under `transcript-poseidon` this is the live
// 128-bit challenge path (exercised by jolt-eval's Poseidon transcript invariant).
// See the fuller note above the prover-side Poseidon impl in `prover.rs`. Do NOT
// delete it.
#[cfg(feature = "transcript-poseidon")]
impl OptimizedChallenge for VerifierState<'_, crate::PoseidonSponge> {
    fn challenge_u128(&mut self) -> u128 {
        // Mirror of the prover impl: squeeze a full element, keep the low 128 bits.
        crate::poseidon::low_128_bits(VerifierState::verifier_message::<ark_bn254::Fr>(self))
    }
}
