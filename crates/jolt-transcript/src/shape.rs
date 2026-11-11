//! A transcript for learning a protocol's shape without its proof of work.

use spongefish::{DuplexSpongeInit, Encoding, VerificationError};

use crate::transcript::{draw_challenges, sealed::Sealed};
use crate::{
    protocol_tag, verifier_transcript, ChallengeCodec, ChallengeSetSize, FieldTranscript,
    VerifierState,
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
    state: VerifierState<'static, H>,
}

impl<H: DuplexSpongeInit<U = u8>> ShapeTranscript<H> {
    /// A fresh shape transcript.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: verifier_transcript::<H>(&protocol_tag(b"jolt-shape-replay"), &0u64, &[]),
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
