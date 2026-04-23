#![cfg_attr(feature = "guest", no_std)]

use ark_bn254::Fr;
use ark_ff::PrimeField;

/// Horner polynomial evaluation baseline using pure-software ark-bn254 Fr.
///
/// Same computation and I/O as `bn254-fr-horner-sdk-guest` but without the
/// FieldOp coprocessor — every Fr add/mul goes through arkworks' software
/// path. Used to measure cycle-count delta against the native-field SDK.
#[jolt::provable(
    stack_size = 65536,
    heap_size = 131072,
    max_input_size = 8192,
    max_trace_length = 4194304
)]
fn fr_horner_arkworks(x_limbs: [u64; 4], coeffs: [[u64; 4]; 32]) -> [u64; 4] {
    let x = fr_from_limbs(x_limbs[0], x_limbs[1], x_limbs[2], x_limbs[3]);
    let mut acc = Fr::from(0u64);
    let mut i: usize = 32;
    while i > 0 {
        i -= 1;
        let c = fr_from_limbs(coeffs[i][0], coeffs[i][1], coeffs[i][2], coeffs[i][3]);
        acc = acc * x + c;
    }
    fr_to_limbs(&acc)
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
