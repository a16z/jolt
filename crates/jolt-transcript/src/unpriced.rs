//! A transcript wrapper that declares no proof of work, for tests only.

use jolt_field::Ring;

use crate::{ChallengeSetSize, Transcript};

/// `T` with both challenge sets declared as `2^256`, so every difficulty from
/// [`grind_bits_for_loss`](crate::grind_bits_for_loss) is 0: proofs carry no
/// nonces and the transcript absorbs and squeezes exactly what `T` does
/// without proof of work.
///
/// Tests use it to reach checks that a hand-built proof would otherwise fail
/// at its first proof-of-work step, and to run fields whose challenge sets
/// cannot reach the target. It overstates `T`'s challenge sets, so it gives
/// no soundness and must never back a real proof; the `test-utils` feature
/// that exports it is for dev-dependencies.
#[derive(Clone, Default)]
pub struct UnpricedTranscript<T>(pub T);

impl<T: Transcript> Transcript for UnpricedTranscript<T> {
    type Challenge = T::Challenge;
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::power_of_two(256);
    const SCALAR_CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::power_of_two(256);

    fn new(label: &'static [u8]) -> Self {
        Self(T::new(label))
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.0.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Self::Challenge {
        self.0.challenge()
    }

    fn challenge_scalar(&mut self) -> Self::Challenge {
        self.0.challenge_scalar()
    }

    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        self.0.challenge_vector(len)
    }

    fn challenge_scalar_powers(&mut self, len: usize) -> Vec<Self::Challenge>
    where
        Self::Challenge: Ring,
    {
        self.0.challenge_scalar_powers(len)
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        self.0.challenge_bytes(out);
    }

    fn state(&self) -> [u8; 32] {
        self.0.state()
    }
}
