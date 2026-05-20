#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};

/// MLE point evaluation of the equality polynomial over BN254 Fr using
/// software `ark_bn254::Fr` — no coprocessor. Mirrors
/// `bn254-eqpoly-inline-guest`: same `K = 32`, same LCG, same eq formula.
/// Every Fr add/mul runs as Montgomery arithmetic compiled to RV64IMAC.
#[jolt::provable(
    backend = "modular",
    stack_size = 65536,
    heap_size = 1_048_576,
    max_input_size = 8192,
    max_trace_length = 262_144
)]
fn bn254_eqpoly_native(seed: u64) -> [u64; 4] {
    const K: usize = 32;
    let one = Fr::from(1u64);
    let mut acc = one;
    for i in 0..K as u64 {
        let ri = gen_fr(seed, i);
        let xi = gen_fr(seed, i + K as u64);
        let rixi = ri * xi;
        let one_minus_ri = one - ri;
        let one_minus_xi = one - xi;
        let other = one_minus_ri * one_minus_xi;
        let term = rixi + other;
        acc *= term;
    }
    fr_to_limbs(&acc)
}

#[inline]
fn gen_fr(seed: u64, idx: u64) -> Fr {
    let l0 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(idx);
    let l1 = (seed ^ idx).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    fr_from_limbs([l0, l1, 0, 0])
}

#[inline]
fn fr_from_limbs(limbs: [u64; 4]) -> Fr {
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&bytes)
}

#[inline]
fn fr_to_limbs(fr: &Fr) -> [u64; 4] {
    // BN254 Fr's BigInt is always 4 × u64 = 32 bytes little-endian.
    let bytes = fr.into_bigint().to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let chunk: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        *limb = u64::from_le_bytes(chunk);
    }
    limbs
}
