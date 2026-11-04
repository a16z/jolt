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
/// work and a shape-correct proof with no nonces replays to its end. It has
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

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        if ok {
            Ok(())
        } else {
            Err(VerificationError)
        }
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
