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

/// Low 128 bits (limbs 0–1) of a full field-element squeeze, for the Poseidon
/// [`OptimizedChallenge`](crate::OptimizedChallenge) impls.
///
/// The byte-sponges (Blake2b/Keccak) implement `challenge_128` by squeezing a
/// 16-byte `u128` directly, because every byte they emit is uniform. A
/// `PoseidonSponge` squeeze is `state.into_bigint().to_bytes_le()` — a field
/// element whose top limb is bounded by the BN254 modulus, so its *high* bytes
/// are NOT uniform. Reading a 16-byte `u128` off a second, buffered squeeze
/// would therefore draw from those non-uniform high bytes and correlate
/// consecutive challenges. Squeezing a whole element instead (the caller passes
/// `verifier_message::<ark_bn254::Fr>()`) yields uniform low 128 bits and advances the
/// sponge by one fresh permutation per challenge — matching the per-challenge
/// independence the byte-sponges get for free. The low 128 bits of a uniform
/// field element are uniform to within ≈2⁻¹²⁶ (statistical distance ≈ 2¹²⁸/p).
///
/// Returns a `u128` so the caller wraps it with `Fr::from(..)`, exactly as the
/// byte-sponge impls wrap `verifier_message::<u128>()`.
pub(crate) fn low_128_bits(squeezed: Fr) -> u128 {
    let limbs = squeezed.into_bigint().0;
    (u128::from(limbs[1]) << 64) | u128::from(limbs[0])
}

// TODO(#1455 follow-up): `PoseidonSponge` is NOT a proper DSFS sponge. Per mmaker's #1455 review,
// the squeeze is hash-chain-like (not a true sponge), absorb throws away ~1/4 of the rate and
// squeeze wastes ~all of it (≈25% / ≥50% throughput loss), and it is not streaming-friendly. It
// should be rewritten as a real `spongefish::Permutation` consumed via `DuplexSponge<P>`.
// RISK of using it as-is (decision C2 — "include Poseidon now"): any challenge/proof format pinned
// on this sponge — especially under the NARG proof model, where the absorb layout bakes into the
// proof bytes — will break a SECOND time when the sponge is rewritten. Kept now for feature parity;
// revisit before the gnark/on-chain verifier depends on the Poseidon layout.
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
}
