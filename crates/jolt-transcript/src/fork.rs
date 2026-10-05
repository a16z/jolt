//! Seeded forks: off-transcript search keyed by the live transcript.

use crate::Sponge;

/// Byte length of the seed a fork is keyed by.
pub const FORK_SEED_LEN: usize = 32;

/// Domain tag separating every fork from every transcript sponge, which
/// starts by absorbing a 64-byte protocol id.
const FORK_TAG: &[u8] = b"jolt-transcript/fork/v1";

/// A fresh sponge keyed by a seed squeezed from the live transcript and a
/// prover-chosen counter.
///
/// A prover that must search (proof of work, or Fiat-Shamir with aborts) squeezes
/// one [`FORK_SEED_LEN`]-byte seed, tries counters on forks, then sends the
/// accepted counter as a `u32` prover message. The verifier squeezes the same
/// seed, receives the counter, and rebuilds the same fork. The seed binds the
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
