//! Spongefish-native [`VerifierTranscript`] surface.

use ark_bn254::Fr;
use spongefish::{
    Decoding, DuplexSpongeInterface, Encoding, NargDeserialize, VerificationResult, VerifierState,
};

use crate::codec::FieldElOptimized;
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
    fn challenge_128(&mut self) -> Fr {
        VerifierState::verifier_message::<FieldElOptimized>(self).0
    }
}

#[cfg(feature = "transcript-keccak")]
impl OptimizedChallenge for VerifierState<'_, spongefish::instantiations::Keccak> {
    fn challenge_128(&mut self) -> Fr {
        VerifierState::verifier_message::<FieldElOptimized>(self).0
    }
}
