#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_bn254_fr::Fr;

/// Horner polynomial evaluation benchmark using the native-field coprocessor.
///
/// Computes `P(x) = c_0 + c_1·x + c_2·x² + ... + c_{N-1}·x^{N-1}` via
/// Horner's method over BN254 Fr. This is a realistic primitive used in
/// KZG commitment opening verification, Plonk verifier polynomial
/// evaluation, and IPA inner-product arguments.
///
/// N = 32 here (limited by serde's 32-element `Serialize` impl for fixed
/// arrays on the host side): 32 Fr multiplications + 32 Fr additions per
/// evaluation. Input `coeffs` is `[[u64; 4]; 32]` with c_0 first. Output
/// is `P(x)` as four Fr limbs.
#[jolt::provable(
    stack_size = 65536,
    heap_size = 131072,
    max_input_size = 8192,
    max_trace_length = 262144
)]
fn fr_horner_sdk(x_limbs: [u64; 4], coeffs: [[u64; 4]; 32]) -> [u64; 4] {
    let x = Fr::from_limbs(x_limbs);
    let mut acc = Fr::zero();
    // Horner: for i in (0..N).rev() { acc = acc * x + c[i]; }
    let mut i: usize = 32;
    while i > 0 {
        i -= 1;
        let c = Fr::from_limbs(coeffs[i]);
        acc = acc.mul(&x).add(&c);
    }
    acc.to_limbs()
}
