#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_bn254_fr::Fr;

/// MLE point evaluation of the equality polynomial over BN254 Fr using the
/// inline FR coprocessor.
///
/// Computes `eq(r, x) = ∏ᵢ (rᵢ·xᵢ + (1−rᵢ)·(1−xᵢ))` for `r, x ∈ Fr^K`
/// of length `K = 32`. Both vectors are generated deterministically from
/// `seed` via a simple LCG (matches `bn254-eqpoly-native-guest`), so the
/// inline and native backends produce identical outputs.
///
/// Workload per term: 2 muls, 2 subs, 1 add, plus 1 final-product mul.
/// Total: `K · 5 + K = 6K = 192` Fr ops, which exercises tight chains of
/// `Fr::mul / Fr::sub / Fr::add` against the coprocessor without any
/// Poseidon-style round constants or MDS structure to muddy the signal.
/// K is sized so the software-Fr native backend still fits within the
/// fixture's 2^18 trace ceiling.
#[jolt::provable(
    backend = "modular",
    stack_size = 65536,
    heap_size = 131072,
    max_input_size = 8192,
    max_trace_length = 262_144
)]
fn bn254_eqpoly_inline(seed: u64) -> [u64; 4] {
    const K: usize = 32;
    let one = Fr::one();
    let mut acc = one;
    for i in 0..K as u64 {
        let ri = gen_fr(seed, i);
        let xi = gen_fr(seed, i + K as u64);
        // term = ri·xi + (1−ri)·(1−xi)
        let rixi = ri.mul(&xi);
        let one_minus_ri = one.sub(&ri);
        let one_minus_xi = one.sub(&xi);
        let other = one_minus_ri.mul(&one_minus_xi);
        let term = rixi.add(&other);
        acc = acc.mul(&term);
    }
    acc.to_limbs()
}

/// Deterministic Fr generator shared between inline and native backends.
/// Returns a canonical Fr (high limbs zero → value < 2^128 < p) so that
/// both `SdkFr::from_limbs` (no reduction) and ark's `Fr::from_le_bytes_mod_order`
/// produce the same logical field element.
#[inline]
fn gen_fr(seed: u64, idx: u64) -> Fr {
    let l0 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(idx);
    let l1 = (seed ^ idx).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    Fr::from_limbs([l0, l1, 0, 0])
}
