//! Look-ahead over a copy of the public sponge state.

use jolt_field::CanonicalBytes;

use crate::grinding::encode_nonce;
use crate::Sponge;

/// A detached copy of a transcript's sponge.
///
/// A preview absorbs and squeezes exactly as the live transcript would, but it
/// holds no argument string and no reference to the transcript, so it cannot
/// change the proof or advance the live sponge. Proof-of-work searches use it
/// to test candidate nonces before committing the winner.
#[derive(Clone, Debug)]
pub struct Preview<H>(H);

impl<H: Sponge> Preview<H> {
    pub(crate) fn new(sponge: H) -> Self {
        Self(sponge)
    }

    /// Absorbs `value` as a prover message or public atom would be absorbed.
    pub fn absorb<A: CanonicalBytes>(&mut self, value: &A) {
        let _ = self.0.absorb(&value.to_bytes_le_vec());
    }

    /// Absorbs raw bytes, as an exact-length byte message would be absorbed.
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        let _ = self.0.absorb(bytes);
    }

    /// Absorbs `nonce` in the encoding [`send_nonce`](crate::ProverTranscript::send_nonce) uses.
    pub fn absorb_nonce(&mut self, nonce: u32) {
        let _ = self.0.absorb(encode_nonce(nonce).as_slice());
    }

    /// Squeezes `N` bytes, as a challenge would.
    #[must_use]
    pub fn squeeze<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        let _ = self.0.squeeze(&mut out);
        out
    }
}
