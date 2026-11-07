//! A transcript for learning a protocol's shape without its proof of work.

use spongefish::{DuplexSpongeInit, Encoding, VerificationError};

use crate::transcript::{draw_challenges, sealed::Sealed};
use crate::{
    protocol_tag, prover_transcript, ChallengeCodec, ChallengeSetSize, FieldTranscript, ProverState,
};

/// A transcript a verifier replays to count what a proof's messages take,
/// such as a masked proof's pad schedule, over messages that are only
/// shape-correct.
///
/// It absorbs into and draws challenges from its own throwaway sponge `H`,
/// and it prices every challenge set at 2^256, so no step asks for proof of
/// work. It passes every check, so a proof's values and nonces do not stop
/// the replay: a shape-correct proof replays to its end whether or not it
/// carries the nonces a real transcript's prices ask for. It has
/// no NARG string and implements neither [`VerifierInit`](crate::VerifierInit)
/// nor [`ProverInit`](crate::ProverInit), so no proof can be verified or
/// produced with it.
pub struct ShapeTranscript<H: DuplexSpongeInit<U = u8>> {
    state: ProverState<H>,
}

impl<H: DuplexSpongeInit<U = u8>> ShapeTranscript<H> {
    /// A fresh shape transcript.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: prover_transcript::<H>(&protocol_tag(b"jolt-shape-replay"), &0u64),
        }
    }
}

impl<H: DuplexSpongeInit<U = u8>> Default for ShapeTranscript<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: DuplexSpongeInit<U = u8>> Sealed for ShapeTranscript<H> {
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        self.state.absorb(message);
    }

    fn squeeze<T>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> T) -> T {
        self.state.squeeze(len, decode)
    }

    fn require(&mut self, _ok: bool) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl<H, F> FieldTranscript<F> for ShapeTranscript<H>
where
    H: DuplexSpongeInit<U = u8>,
    F: ChallengeCodec,
{
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::power_of_two(256);

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        draw_challenges(self, len)
    }
}

#[cfg(all(test, feature = "bn254", feature = "transcript-keccak"))]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use jolt_field::Fr;
    use spongefish::instantiations::Keccak;

    use super::ShapeTranscript;
    use crate::{FieldTranscript, Transcript, MAX_POW_BITS};

    /// Every proof-of-work step passes, with any nonce or none, and the
    /// replay goes on drawing challenges.
    #[test]
    fn every_check_passes_and_draws_continue() {
        let mut shape = ShapeTranscript::<Keccak>::new();
        shape.append(&7u64);
        shape.check_public_grind(MAX_POW_BITS, Some(0)).unwrap();
        shape.check_public_grind(MAX_POW_BITS, None).unwrap();
        shape.check_public_grind(0, Some(3)).unwrap();
        let first: Fr = shape.challenge().unwrap();
        let second: Fr = shape.challenge().unwrap();
        assert_ne!(first, second);
    }

    /// The draws depend only on what was absorbed, so two replays of the
    /// same shape agree.
    #[test]
    fn replays_of_one_shape_agree() {
        let draw = || {
            let mut shape = ShapeTranscript::<Keccak>::new();
            shape.append(&11u64);
            FieldTranscript::<Fr>::challenge_vector(&mut shape, 3).unwrap()
        };
        assert_eq!(draw(), draw());
    }
}
