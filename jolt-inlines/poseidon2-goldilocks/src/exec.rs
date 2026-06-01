// SPDX-License-Identifier: Apache-2.0

//! Host-side reference implementation of the Goldilocks Poseidon2
//! permutation.
//!
//! This is the ground-truth comparator the sequence builder must
//! match byte-for-byte. The tests compare it against Plonky3's
//! canonical `Poseidon2Goldilocks<8>` implementation.

use crate::{Poseidon2GoldilocksState, GOLDILOCKS_MODULUS, STATE_WIDTH};

// Re-exported from `crate` root so existing callers can keep their
// `crate::exec::POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8` and
// `crate::exec::add_mod` imports. The canonical definitions live in
// `lib.rs` because the SDK guest path needs them in `no_std` builds.
pub use crate::{add_mod, POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8};

/// Diagonal matrix for the internal-round diffusion step.
#[rustfmt::skip]
pub const POSEIDON2_INTERNAL_DIAG: [u64; STATE_WIDTH] = [
    0xfffffffeffffffff,
    1,
    2,
    0x7fffffff80000001,
    3,
    0x7fffffff80000000,
    0xfffffffefffffffe,
    0xfffffffefffffffd,
];

#[inline]
pub fn mul_mod(a: u64, b: u64) -> u64 {
    let res = (a as u128) * (b as u128);
    let lo = res as u64;
    let hi = (res >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi as u32 as u64;

    // `add_term` is `lo + (hi_lo << 32)`. This sum can exceed 2^64.
    // The naive wrapping_add loses 2^64 worth of magnitude in that
    // case — and since 2^64 ≡ (2^32 - 1) mod P, the result is short
    // by (2^32 - 1) mod P when the overflow happens. Detect and
    // compensate.
    let (add_term, add_overflow) = lo.overflowing_add(hi_lo << 32);
    let sub_term = hi_lo + hi_hi;

    let mut r = add_term.wrapping_sub(sub_term);
    if add_term < sub_term {
        r = r.wrapping_add(GOLDILOCKS_MODULUS);
    }

    if add_overflow {
        // Add (2^32 - 1) to recover the lost magnitude. If THIS add
        // overflows u64, the wrap is equivalent to subtracting another
        // 2^64 ≡ (2^32 - 1) mod P from the result — so we add
        // (2^32 - 1) one more time.
        let (r1, wrapped) = r.overflowing_add(0xFFFFFFFF);
        r = r1;
        if wrapped {
            r = r.wrapping_add(0xFFFFFFFF);
        }
    }

    while r >= GOLDILOCKS_MODULUS {
        r -= GOLDILOCKS_MODULUS;
    }
    r
}

/// S-box: `x^7` over Goldilocks. Computed as `x^4 * x^2 * x` (3 mults).
#[inline]
pub fn sbox(x: u64) -> u64 {
    let x2 = mul_mod(x, x);
    let x4 = mul_mod(x2, x2);
    let x3 = mul_mod(x2, x);
    mul_mod(x4, x3)
}

/// External MDS layer: 8-wide matrix multiply via two m4 sub-blocks
/// plus the cross-mixing step.
pub fn external_mds(state: &mut [u64; STATE_WIDTH]) {
    fn m4(s: &mut [u64]) {
        let (a, b, c, d) = (s[0], s[1], s[2], s[3]);
        let sum = add_mod(add_mod(a, b), add_mod(c, d));
        s[0] = add_mod(sum, add_mod(a, add_mod(b, b)));
        s[1] = add_mod(sum, add_mod(b, add_mod(c, c)));
        s[2] = add_mod(sum, add_mod(c, add_mod(d, d)));
        s[3] = add_mod(sum, add_mod(d, add_mod(a, a)));
    }
    let mut left = [state[0], state[1], state[2], state[3]];
    let mut right = [state[4], state[5], state[6], state[7]];
    m4(&mut left);
    m4(&mut right);
    for i in 0..4 {
        state[i] = add_mod(left[i], right[i]);
        state[i + 4] = add_mod(left[i], right[i]);
    }
    for i in 0..4 {
        state[i] = add_mod(state[i], left[i]);
        state[i + 4] = add_mod(state[i + 4], right[i]);
    }
}

/// Internal-round diffusion: multiply by diagonal, then add row-sum
/// to every coordinate.
pub fn internal_diffusion(state: &mut [u64; STATE_WIDTH]) {
    let mut sum = 0;
    for &s in state.iter() {
        sum = add_mod(sum, s);
    }
    for i in 0..STATE_WIDTH {
        state[i] = add_mod(mul_mod(POSEIDON2_INTERNAL_DIAG[i], state[i]), sum);
    }
}

/// The Poseidon2 permutation in full.
pub fn execute_poseidon2_permutation(state: &mut Poseidon2GoldilocksState) {
    let mut rc_idx = 0;

    external_mds(state);

    // 4 external initial rounds
    for _ in 0..4 {
        for s in state.iter_mut() {
            *s = add_mod(*s, POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[rc_idx]);
            rc_idx += 1;
        }
        for s in state.iter_mut() {
            *s = sbox(*s);
        }
        external_mds(state);
    }

    // 22 internal rounds
    for _ in 0..22 {
        state[0] = add_mod(state[0], POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[rc_idx]);
        rc_idx += 1;
        state[0] = sbox(state[0]);
        internal_diffusion(state);
    }

    // 4 external final rounds
    for _ in 0..4 {
        for s in state.iter_mut() {
            *s = add_mod(*s, POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8[rc_idx]);
            rc_idx += 1;
        }
        for s in state.iter_mut() {
            *s = sbox(*s);
        }
        external_mds(state);
    }
}
