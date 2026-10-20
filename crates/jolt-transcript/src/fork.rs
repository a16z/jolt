//! Seeded forks: off-transcript search keyed by the live transcript.

use crate::Sponge;

/// Byte length of the seed a fork is keyed by.
pub const FORK_SEED_LEN: usize = 32;

/// Byte length of the secret nonce an attempt fork is keyed by.
pub const ATTEMPT_NONCE_LEN: usize = 32;

/// Domain tag separating every fork from every transcript sponge, which
/// starts by absorbing a 64-byte protocol id.
const FORK_TAG: &[u8] = b"jolt-transcript/fork/v1";
/// Domain tag of attempt forks, which the random-oracle analysis of a
/// Fiat–Shamir-with-aborts step treats as a separate oracle.
const ATTEMPT_TAG: &[u8] = b"jolt-transcript/fork/attempt/v1";

/// A fresh sponge keyed by a seed squeezed from the live transcript and a
/// prover-chosen counter, or for [`Self::attempt`] a secret nonce.
///
/// A prover that must search for proof of work squeezes one
/// [`FORK_SEED_LEN`]-byte seed, tries counters on forks, then sends the
/// accepted counter as a canonical LEB128 [`Nonce`](crate::Nonce) message.
/// The verifier squeezes the same seed, receives the counter, and rebuilds
/// the same fork. The seed binds the
/// whole prechallenge transcript, and the counter is in the argument string,
/// so nothing a fork derives can be chosen independently of the transcript.
/// No transcript state is ever copied.
#[derive(Clone, Debug)]
pub struct Fork<H>(H);

impl<H: Sponge> Fork<H> {
    /// The fork for `counter` under `seed`.
    #[must_use]
    pub fn new(seed: &[u8; FORK_SEED_LEN], counter: u32) -> Self {
        let mut sponge = H::default();
        let _ = sponge.absorb(FORK_TAG);
        let _ = sponge.absorb(seed);
        let _ = sponge.absorb(&counter.to_le_bytes());
        Self(sponge)
    }

    /// The fork for one attempt of a Fiat–Shamir-with-aborts step, keyed by
    /// `seed` and a secret nonce.
    ///
    /// Unlike a grinding counter, the nonce must be drawn uniformly from the
    /// prover's RNG for every attempt and stays secret until the prover sends
    /// it with the accepted attempt: zero knowledge needs every rejected
    /// attempt's oracle input to be unpredictable, so that its answer is never
    /// observed (Whiteout, Thm `wo:thm:zk`, hybrid H2). The caller absorbs
    /// the attempt's first message before squeezing its challenge.
    #[must_use]
    pub fn attempt(seed: &[u8; FORK_SEED_LEN], nonce: &[u8; ATTEMPT_NONCE_LEN]) -> Self {
        let mut sponge = H::default();
        let _ = sponge.absorb(ATTEMPT_TAG);
        let _ = sponge.absorb(seed);
        let _ = sponge.absorb(nonce);
        Self(sponge)
    }

    /// Absorbs `bytes`.
    pub fn absorb(&mut self, bytes: &[u8]) {
        let _ = self.0.absorb(bytes);
    }

    /// Squeezes `N` bytes.
    #[must_use]
    pub fn squeeze<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        let _ = self.0.squeeze(&mut out);
        out
    }
}

#[cfg(all(test, feature = "transcript-blake2b"))]
mod tests {
    use super::*;
    use spongefish::instantiations::Blake2b512;

    fn attempt_challenge(seed: [u8; FORK_SEED_LEN], nonce: [u8; ATTEMPT_NONCE_LEN]) -> [u8; 32] {
        let mut fork = Fork::<Blake2b512>::attempt(&seed, &nonce);
        fork.absorb(b"first message");
        fork.squeeze()
    }

    #[test]
    fn attempt_forks_are_keyed_by_seed_and_nonce() {
        let base = attempt_challenge([1; 32], [2; 32]);
        assert_eq!(base, attempt_challenge([1; 32], [2; 32]));
        assert_ne!(base, attempt_challenge([3; 32], [2; 32]));
        let mut nonce = [2; 32];
        nonce[31] ^= 1;
        assert_ne!(base, attempt_challenge([1; 32], nonce));
    }

    #[test]
    fn attempt_forks_are_separated_from_grinding_forks() {
        // A grinding fork absorbs its counter's little-endian bytes after the
        // seed; a nonce with the same leading bytes must still give an
        // unrelated stream.
        let seed = [5; 32];
        let mut nonce = [0; ATTEMPT_NONCE_LEN];
        nonce[..4].copy_from_slice(&7u32.to_le_bytes());
        let mut grinding = Fork::<Blake2b512>::new(&seed, 7);
        let mut attempt = Fork::<Blake2b512>::attempt(&seed, &nonce);
        assert_ne!(grinding.squeeze::<32>(), attempt.squeeze::<32>());
    }
}
