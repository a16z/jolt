//! `PoseidonSponge` — Circom-compatible BN254 Poseidon adapter exposed
//! through `spongefish::DuplexSpongeInterface`.
//!
//! Sponge layout: one `Fr` capacity element (`self.state`) plus two `Fr`
//! rate inputs per `permute` call, fed through light-poseidon's width-4
//! compression function (`Poseidon::new_circom(3)` — width minus one
//! inputs). Each call replaces capacity with the compression output.
//!
//! Byte traffic is mapped to `Fr` via 31-byte little-endian chunks
//! (`Fr::from_le_bytes_mod_order` is injective on chunks ≤ 31 bytes since
//! 248 bits < BN254 modulus). Squeezed bytes come from
//! `into_bigint().to_bytes_le()` of the running state.
//!
//! Round constants are built once per `PoseidonSponge` construction; the
//! same `Poseidon<Fr>` is reused for every `permute` call in this sponge's
//! lifetime (`light_poseidon::Poseidon::hash` clears its scratch state on
//! exit, so reuse is safe).

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField, Zero};
use light_poseidon::{Poseidon, PoseidonHasher};
use spongefish::DuplexSpongeInterface;

const SQUEEZE_BYTES: usize = 32;
const ABSORB_CHUNK_BYTES: usize = 31;

#[expect(
    clippy::expect_used,
    reason = "width 4 (NR_INPUTS=3) is supported by light-poseidon's Circom params"
)]
fn fresh_hasher() -> Poseidon<Fr> {
    Poseidon::<Fr>::new_circom(3).expect("light-poseidon: width-4 init")
}

/// Width-4 Poseidon duplex sponge over BN254 `Fr`, byte-driven.
pub struct PoseidonSponge {
    hasher: Poseidon<Fr>,
    state: Fr,
    pending_squeeze: [u8; SQUEEZE_BYTES],
    squeeze_offset: usize,
}

impl PoseidonSponge {
    /// Construct a fresh sponge with zero state.
    pub fn new() -> Self {
        Self {
            hasher: fresh_hasher(),
            state: Fr::zero(),
            pending_squeeze: [0u8; SQUEEZE_BYTES],
            squeeze_offset: SQUEEZE_BYTES,
        }
    }

    /// One Poseidon application over the current state plus two `Fr` rate
    /// inputs.
    fn permute(&mut self, a: Fr, b: Fr) {
        #[expect(
            clippy::expect_used,
            reason = "input length matches hasher width-1; failure is unreachable"
        )]
        let next = self
            .hasher
            .hash(&[self.state, a, b])
            .expect("light-poseidon hash");
        self.state = next;
    }

    fn refill_squeeze(&mut self) {
        self.permute(Fr::zero(), Fr::zero());
        let bytes = self.state.into_bigint().to_bytes_le();
        self.pending_squeeze.fill(0);
        let n = bytes.len().min(SQUEEZE_BYTES);
        self.pending_squeeze[..n].copy_from_slice(&bytes[..n]);
        self.squeeze_offset = 0;
    }

    fn absorb_fr_pair(&mut self, a: Fr, b: Fr) {
        // Any pending squeeze is invalidated by a new absorb; spongefish's
        // DuplexSpongeInterface contract is associative within a phase and
        // the squeeze cache is just a buffer over fresh permutations.
        self.squeeze_offset = SQUEEZE_BYTES;
        self.permute(a, b);
    }
}

impl Default for PoseidonSponge {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PoseidonSponge {
    fn clone(&self) -> Self {
        Self {
            hasher: fresh_hasher(),
            state: self.state,
            pending_squeeze: self.pending_squeeze,
            squeeze_offset: self.squeeze_offset,
        }
    }
}

impl std::fmt::Debug for PoseidonSponge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseidonSponge")
            .field("state", &self.state)
            .field("squeeze_offset", &self.squeeze_offset)
            .finish_non_exhaustive()
    }
}

impl DuplexSpongeInterface for PoseidonSponge {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        // Length-binding permutation up front: without it, absorb(&[]),
        // absorb(&[0]), absorb(&[0; 31]) all collapse to Fr::zero() at the
        // chunk level and would alias.
        let len_fr = Fr::from(input.len() as u64);
        self.absorb_fr_pair(len_fr, Fr::zero());

        let mut iter = input.chunks(ABSORB_CHUNK_BYTES);
        while let Some(first) = iter.next() {
            let a = Fr::from_le_bytes_mod_order(first);
            let b = iter
                .next()
                .map_or_else(Fr::zero, Fr::from_le_bytes_mod_order);
            self.absorb_fr_pair(a, b);
        }
        self
    }

    fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        let mut written = 0;
        while written < output.len() {
            if self.squeeze_offset >= SQUEEZE_BYTES {
                self.refill_squeeze();
            }
            let avail = SQUEEZE_BYTES - self.squeeze_offset;
            let want = output.len() - written;
            let take = avail.min(want);
            output[written..written + take].copy_from_slice(
                &self.pending_squeeze[self.squeeze_offset..self.squeeze_offset + take],
            );
            self.squeeze_offset += take;
            written += take;
        }
        self
    }

    fn ratchet(&mut self) -> &mut Self {
        // Intentional double-permute on the `ratchet ; squeeze` path: the
        // permutation here is the one-way ratchet, and the next `squeeze`
        // call's `refill_squeeze` runs a second permutation to produce the
        // first squeeze block. Matches spongefish's own `DuplexSponge::ratchet`
        // semantics (one permute in `ratchet`, another in the first
        // `squeeze`); do not collapse to a single call.
        self.permute(Fr::zero(), Fr::zero());
        self.squeeze_offset = SQUEEZE_BYTES;
        self
    }
}

#[cfg(test)]
#[expect(
    unused_results,
    reason = "DuplexSpongeInterface methods return &mut Self for chaining"
)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(b"hello");
        b.absorb(b"hello");
        let mut x = [0u8; 64];
        let mut y = [0u8; 64];
        a.squeeze(&mut x);
        b.squeeze(&mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn order_sensitive() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(b"x").absorb(b"y");
        b.absorb(b"y").absorb(b"x");
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        a.squeeze(&mut x);
        b.squeeze(&mut y);
        assert_ne!(x, y);
    }

    #[test]
    fn empty_distinct_from_zero_absorb() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(&[]);
        b.absorb(&[0u8]);
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        a.squeeze(&mut x);
        b.squeeze(&mut y);
        assert_ne!(x, y);
    }

    /// Clone rebuilds the (non-Clone) hasher but must preserve the duplex
    /// state, including a partially consumed squeeze block.
    #[test]
    fn clone_resumes_a_partially_consumed_squeeze_block() {
        let mut original = PoseidonSponge::new();
        original.absorb(b"clone me");
        let mut prefix = [0u8; 11];
        original.squeeze(&mut prefix);

        let mut cloned = original.clone();
        let mut rest_original = [0u8; 40];
        let mut rest_cloned = [0u8; 40];
        original.squeeze(&mut rest_original);
        cloned.squeeze(&mut rest_cloned);
        assert_eq!(rest_original, rest_cloned);
    }

    #[test]
    fn clones_diverge_after_independent_absorbs() {
        let mut left = PoseidonSponge::new();
        left.absorb(b"shared");
        let mut right = left.clone();
        left.absorb(b"left");
        right.absorb(b"right");
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        left.squeeze(&mut x);
        right.squeeze(&mut y);
        assert_ne!(x, y);
    }

    #[test]
    fn debug_output_reports_the_squeeze_offset() {
        let sponge = PoseidonSponge::new();
        let output = format!("{sponge:?}");
        assert!(
            output.contains("PoseidonSponge") && output.contains("squeeze_offset"),
            "unexpected Debug output: {output}"
        );
    }

    #[test]
    fn ratchet_advances_the_state_and_discards_pending_output() {
        let mut baseline_sponge = PoseidonSponge::new();
        baseline_sponge.absorb(b"m");
        let mut baseline = [0u8; 32];
        baseline_sponge.squeeze(&mut baseline);

        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(b"m").ratchet();
        b.absorb(b"m").ratchet();
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        a.squeeze(&mut x);
        b.squeeze(&mut y);
        assert_eq!(x, y, "ratchet must be deterministic");
        assert_ne!(x, baseline, "ratchet must advance the state");

        // A ratchet after a partial squeeze must not resume the pending
        // block: the remaining 27 bytes of the pre-ratchet block would be
        // `baseline[5..]`.
        let mut partial = PoseidonSponge::new();
        partial.absorb(b"m");
        let mut consumed = [0u8; 5];
        partial.squeeze(&mut consumed);
        assert_eq!(consumed, baseline[..5], "fixture must share the block");
        partial.ratchet();
        let mut post_ratchet = [0u8; 27];
        partial.squeeze(&mut post_ratchet);
        assert_ne!(
            post_ratchet[..],
            baseline[5..],
            "pending squeeze bytes must be discarded by ratchet"
        );
    }
}
