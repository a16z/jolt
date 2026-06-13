//! `PoseidonSponge` — Circom-compatible BN254 Poseidon adapter exposed
//! through `spongefish::DuplexSpongeInterface` with **`U = Fr`** (the
//! field-aligned transcript of `specs/transpiler-optimization-spec.md` §4).
//!
//! Sponge layout (unchanged compression chain, spec decision D2): one `Fr`
//! capacity element (`self.state`) plus two `Fr` rate inputs per `permute`
//! call, fed through light-poseidon's width-4 compression function
//! (`Poseidon::new_circom(3)` — width minus one inputs). Each call replaces
//! capacity with the compression output. This is a hash chain, not a true
//! duplex sponge (no hidden capacity across squeezes); cost and
//! ideal-permutation uniformity are unaffected (spec §11.2).
//!
//! - `absorb(&[Fr])` feeds unit pairs: `permute(u0,u1)`, `permute(u2,u3)`, …
//!   zero-padding an odd tail. Every absorb call starts a fresh permute pair,
//!   so message boundaries bind exactly when each message is a complete
//!   tagged group per spec §4.2 (the tagged-length encoding lives in the
//!   message layer — see [`crate::codec`] — NOT here; the sponge just eats
//!   unit slices).
//! - `squeeze(&mut [Fr])`: per output unit, `permute(0,0)` and emit the new
//!   state — one permute per squeezed unit, no buffering.
//!
//! Round constants are built once per `PoseidonSponge` construction; the
//! same `Poseidon<Fr>` is reused for every `permute` call in this sponge's
//! lifetime (`light_poseidon::Poseidon::hash` clears its scratch state on
//! exit, so reuse is safe).

use ark_bn254::Fr;
use ark_ff::Zero;
use light_poseidon::{Poseidon, PoseidonHasher};
use spongefish::DuplexSpongeInterface;

#[expect(
    clippy::expect_used,
    reason = "width 4 (NR_INPUTS=3) is supported by light-poseidon's Circom params"
)]
fn fresh_hasher() -> Poseidon<Fr> {
    Poseidon::<Fr>::new_circom(3).expect("light-poseidon: width-4 init")
}

/// Width-4 Poseidon compression-chain sponge over BN254 `Fr`, field-unit
/// driven (`U = Fr`).
pub struct PoseidonSponge {
    hasher: Poseidon<Fr>,
    state: Fr,
}

impl PoseidonSponge {
    /// Construct a fresh sponge with zero state.
    pub fn new() -> Self {
        Self {
            hasher: fresh_hasher(),
            state: Fr::zero(),
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
        }
    }
}

impl std::fmt::Debug for PoseidonSponge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseidonSponge")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl DuplexSpongeInterface for PoseidonSponge {
    type U = Fr;

    fn absorb(&mut self, input: &[Fr]) -> &mut Self {
        for pair in input.chunks(2) {
            let a = pair[0];
            let b = pair.get(1).copied().unwrap_or_else(Fr::zero);
            self.permute(a, b);
        }
        self
    }

    fn squeeze(&mut self, output: &mut [Fr]) -> &mut Self {
        for slot in output {
            self.permute(Fr::zero(), Fr::zero());
            *slot = self.state;
        }
        self
    }

    fn ratchet(&mut self) -> &mut Self {
        self.permute(Fr::zero(), Fr::zero());
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

    fn squeeze1(s: &mut PoseidonSponge) -> Fr {
        let mut out = [Fr::zero(); 1];
        s.squeeze(&mut out);
        out[0]
    }

    #[test]
    fn deterministic() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        let units = [Fr::from(7u64), Fr::from(11u64), Fr::from(13u64)];
        a.absorb(&units);
        b.absorb(&units);
        assert_eq!(squeeze1(&mut a), squeeze1(&mut b));
    }

    #[test]
    fn order_sensitive() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(&[Fr::from(1u64), Fr::from(2u64)]);
        b.absorb(&[Fr::from(2u64), Fr::from(1u64)]);
        assert_ne!(squeeze1(&mut a), squeeze1(&mut b));
    }

    /// One squeezed unit costs exactly one permute: two single-unit squeezes
    /// equal one two-unit squeeze (associativity, no buffering).
    #[test]
    fn squeeze_is_associative_one_permute_per_unit() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        let x = squeeze1(&mut a);
        let y = squeeze1(&mut a);
        let mut out = [Fr::zero(); 2];
        b.squeeze(&mut out);
        assert_eq!([x, y], out);
    }

    /// Odd-length absorbs zero-pad the trailing pair.
    #[test]
    fn odd_absorb_pads_with_zero_unit() {
        let mut a = PoseidonSponge::new();
        let mut b = PoseidonSponge::new();
        a.absorb(&[Fr::from(5u64)]);
        b.absorb(&[Fr::from(5u64), Fr::zero()]);
        assert_eq!(squeeze1(&mut a), squeeze1(&mut b));
    }
}
