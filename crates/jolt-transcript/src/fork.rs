//! Seeded forks: off-transcript search keyed by the live transcript.

use std::fmt::{Debug, Formatter, Result as FmtResult};

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
#[derive(Clone)]
pub struct Fork<H>(H);

/// Redacted: an attempt fork is keyed by a secret nonce, so its sponge state
/// is never printed.
impl<H> Debug for Fork<H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("Fork(..)")
    }
}

/// Whether `first_absorb`, the first bytes a sponge absorbs, starts a fork
/// rather than a transcript. Transcript audits use it to tell the two apart.
#[must_use]
pub fn is_fork_domain(first_absorb: &[u8]) -> bool {
    first_absorb == FORK_TAG || first_absorb == ATTEMPT_TAG
}

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
#[expect(clippy::indexing_slicing, reason = "test code")]
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

    /// Known answer: the verifier rebuilds this fork from the transcript
    /// seed and the sent nonce, so its bytes are protocol.
    #[test]
    fn attempt_fork_known_answer() {
        let hex = attempt_challenge([1; 32], [2; 32])
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect::<String>();
        assert_eq!(
            hex,
            "6b9654574b422eb9f535a3e36ce659b05b71d458d8d7222060bd6e6c45c54f71"
        );
    }

    #[test]
    fn fork_domains_are_recognized() {
        assert!(is_fork_domain(FORK_TAG));
        assert!(is_fork_domain(ATTEMPT_TAG));
        assert!(!is_fork_domain(b"jolt-transcript/fork"));
    }
}
