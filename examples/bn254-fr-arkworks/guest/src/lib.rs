#![cfg_attr(feature = "guest", no_std)]

use ark_bn254::Fr;
use ark_ff::PrimeField;

/// Baseline comparison: same `(a + b) * a` computation as the
/// `bn254-fr-smoke-guest`, but computed purely through `ark_bn254::Fr`'s
/// software implementation — no FieldOp coprocessor. Used to measure the
/// cycle-count delta vs. the native-field inline SDK.
#[jolt::provable(heap_size = 32768, max_trace_length = 1048576)]
fn fr_add_mul_arkworks(
    a_lo: u64,
    a_1: u64,
    a_2: u64,
    a_3: u64,
    b_lo: u64,
    b_1: u64,
    b_2: u64,
    b_3: u64,
) -> [u64; 4] {
    let a = fr_from_limbs(a_lo, a_1, a_2, a_3);
    let b = fr_from_limbs(b_lo, b_1, b_2, b_3);
    let sum = a + b;
    let prod = sum * a;
    fr_to_limbs(&prod)
}

fn fr_from_limbs(l0: u64, l1: u64, l2: u64, l3: u64) -> Fr {
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&l0.to_le_bytes());
    bytes[8..16].copy_from_slice(&l1.to_le_bytes());
    bytes[16..24].copy_from_slice(&l2.to_le_bytes());
    bytes[24..32].copy_from_slice(&l3.to_le_bytes());
    Fr::from_le_bytes_mod_order(&bytes)
}

fn fr_to_limbs(fr: &Fr) -> [u64; 4] {
    use ark_ff::BigInteger;
    let bi = fr.into_bigint();
    let bytes = bi.to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = core::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}
