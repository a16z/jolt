//! Operations shared by both roles' spongefish states.

use jolt_field::{CanonicalEncoding, Field};
use rand_core::RngCore;
use spongefish::{Encoding, ProverState, VerifierState};

use crate::{ProtocolId, Sponge};

/// Bytes squeezed for a [`Channel::challenge_small`](crate::Channel::challenge_small).
pub(crate) const SMALL_CHALLENGE_BYTES: usize = 16;

/// Width of the squeeze blocks a challenge is drawn in.
const SQUEEZE_BLOCK: usize = 32;

/// Width of the blocks a known-length byte message is sent and read in. Both
/// roles split identically, then the remainder byte by byte, so every sponge
/// sees the same absorb calls on both sides.
pub(crate) const BYTE_BLOCK: usize = 32;

/// A byte string absorbed as `LE64(len) || bytes`, so consecutive
/// variable-length values cannot run into each other.
pub(crate) struct Framed<'a>(pub(crate) &'a [u8]);

impl Encoding<[u8]> for Framed<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::with_capacity(8 + self.0.len());
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        out.extend_from_slice(self.0);
        out
    }
}

/// The spongefish domain separator every Jolt-family transcript starts from:
/// the protocol id, then the framed session, then an empty framed instance.
pub(crate) fn domain<'s>(
    protocol: &ProtocolId,
    session: &'s [u8],
) -> spongefish::DomainSeparator<
    spongefish::WithInstance<Framed<'static>>,
    spongefish::WithSession<Framed<'s>>,
> {
    spongefish::DomainSeparator::new(*protocol.as_bytes())
        .session(Framed(session))
        .instance(Framed(&[]))
}

/// The verifier-message (squeeze) half of a spongefish state.
pub(crate) trait Squeeze {
    /// Squeezes exactly `out.len()` bytes of the byte stream.
    fn squeeze_into(&mut self, out: &mut [u8]);

    fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        self.squeeze_into(&mut out);
        out
    }

    /// An exactly uniform element of `F`, per [`Field::random`]'s
    /// rejection-sampling contract over the squeezed byte stream. Returns the
    /// element and the number of bytes squeezed.
    fn exact_challenge<F: Field>(&mut self) -> (F, usize)
    where
        Self: Sized,
    {
        let mut rng = SqueezeRng {
            state: self,
            squeezed: 0,
        };
        let value = F::random(&mut rng);
        (value, rng.squeezed)
    }

    fn small_challenge<F: CanonicalEncoding>(&mut self) -> F {
        F::from_challenge_bytes(&self.squeeze_array::<SMALL_CHALLENGE_BYTES>())
    }
}

/// Squeezes `out` in whole blocks, then single bytes; both roles squeeze
/// through here, so they make the same squeeze calls.
macro_rules! squeeze_blocks {
    ($state:expr, $out:expr) => {{
        let mut blocks = $out.chunks_exact_mut(SQUEEZE_BLOCK);
        for block in &mut blocks {
            block.copy_from_slice(&$state.verifier_message::<[u8; SQUEEZE_BLOCK]>());
        }
        for byte in blocks.into_remainder() {
            [*byte] = $state.verifier_message::<[u8; 1]>();
        }
    }};
}

impl<H: Sponge> Squeeze for ProverState<H, rand::rngs::StdRng> {
    fn squeeze_into(&mut self, out: &mut [u8]) {
        squeeze_blocks!(self, out);
    }
}

impl<H: Sponge> Squeeze for VerifierState<'_, H> {
    fn squeeze_into(&mut self, out: &mut [u8]) {
        squeeze_blocks!(self, out);
    }
}

/// Feeds squeezed bytes to [`Field::random`].
struct SqueezeRng<'a, S> {
    state: &'a mut S,
    squeezed: usize,
}

impl<S: Squeeze> RngCore for SqueezeRng<'_, S> {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.state.squeeze_into(dest);
        self.squeezed += dest.len();
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "test squeezes stay inside their fixed buffer"
)]
mod tests {
    use super::Squeeze;
    use jolt_field::Fr;

    /// A squeeze stream that replays fixed bytes.
    struct Replayed {
        bytes: Vec<u8>,
        position: usize,
    }

    impl Squeeze for Replayed {
        fn squeeze_into(&mut self, out: &mut [u8]) {
            out.copy_from_slice(&self.bytes[self.position..self.position + out.len()]);
            self.position += out.len();
        }
    }

    /// A candidate at or above the modulus is rejected and the draw resamples
    /// from the next 32 squeezed bytes, so it equals a draw from those bytes
    /// alone.
    #[test]
    fn exact_challenge_rejects_out_of_range_candidates_and_resamples() {
        let accepted: Vec<u8> = (1..=32).collect();
        let mut direct = Replayed {
            bytes: accepted.clone(),
            position: 0,
        };
        let (expected, squeezed): (Fr, usize) = direct.exact_challenge();
        assert_eq!(squeezed, 32);

        let mut resampled = Replayed {
            bytes: [vec![0xff; 32], accepted].concat(),
            position: 0,
        };
        let (value, squeezed): (Fr, usize) = resampled.exact_challenge();
        assert_eq!(squeezed, 64);
        assert_eq!(value, expected);
    }
}
