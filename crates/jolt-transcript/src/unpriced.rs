//! A transcript wrapper that declares no proof of work, for tests only.

#[cfg(feature = "prover-grinding")]
use crate::pow::MessageSearch;
use crate::transcript::{draw_challenges, sealed::Sealed};
use crate::{ChallengeCodec, ChallengeSetSize, FieldTranscript, ProverTranscript};
use spongefish::{Encoding, VerificationError};

/// `T` with its challenge set declared as `2^256`, so every difficulty from
/// [`grind_bits_for_loss`](crate::grind_bits_for_loss) is 0: proofs carry no
/// nonces and the transcript absorbs and squeezes exactly what `T` does
/// without proof of work.
///
/// Tests use it to reach checks that a hand-built proof would otherwise fail
/// at its first proof-of-work step. It overstates `T`'s challenge set, so it
/// gives no soundness and must never back a real proof; the `test-utils`
/// feature that exports it is for dev-dependencies.
pub struct UnpricedTranscript<T>(pub T);

impl<T: Sealed> Sealed for UnpricedTranscript<T> {
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        self.0.absorb(message);
    }

    fn squeeze<U>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> U) -> U {
        self.0.squeeze(len, decode)
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        self.0.require(ok)
    }
}

impl<T: FieldTranscript<F>, F: ChallengeCodec> FieldTranscript<F> for UnpricedTranscript<T> {
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::power_of_two(256);

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges(self, len)
    }
}

impl<T: ProverTranscript> ProverTranscript for UnpricedTranscript<T> {
    type Sponge = T::Sponge;

    fn prover_message<M: Encoding + ?Sized>(&mut self, msg: &M) {
        self.0.prover_message(msg);
    }

    fn narg_string(&self) -> &[u8] {
        self.0.narg_string()
    }

    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<T::Sponge> {
        self.0.message_search()
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.0.grind(bits)
    }

    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u32>, VerificationError> {
        self.0.public_grind(bits)
    }
}
