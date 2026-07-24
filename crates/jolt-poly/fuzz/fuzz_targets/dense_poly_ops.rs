#![no_main]

//! Differential check of `Polynomial` evaluation paths against a naive
//! per-index multilinear-extension reference.

use jolt_field::{Fr, FromPrimitiveInt, ReducingBytes};
use jolt_poly::Polynomial;
use libfuzzer_sys::fuzz_target;

/// Bytes per BN254 scalar window.
const SCALAR_BYTES: usize = 32;

/// Largest variable count whose full encoding (`2^n` coefficients plus the
/// point, one 32-byte window each) fits the runner's 4096-byte input cap;
/// 7 variables would need 4321 bytes.
const MAX_NUM_VARS: usize = 6;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let num_vars = (data[0] as usize % MAX_NUM_VARS) + 1; // 1..=6, all reachable
    let n = 1usize << num_vars;
    if data.len() < 1 + (n + num_vars) * SCALAR_BYTES {
        return;
    }

    let scalar_at = |index: usize| {
        let start = 1 + index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let evals: Vec<Fr> = (0..n).map(scalar_at).collect();
    let point: Vec<Fr> = (0..num_vars).map(|i| scalar_at(n + i)).collect();

    let poly = Polynomial::new(evals.clone());

    // The two optimized paths must agree with each other...
    let eval = poly.evaluate(&point);
    let eval_consumed = poly.clone().evaluate_and_consume(&point);
    assert_eq!(
        eval, eval_consumed,
        "evaluate and evaluate_and_consume disagree"
    );

    // ...and with a naive Σᵢ evals[i]·eq(bits(i), point) reference computed
    // without the shared eq-table machinery. The first point coordinate
    // corresponds to the most significant index bit.
    let one = Fr::from_u64(1);
    let mut reference = Fr::from_u64(0);
    for (i, &coeff) in evals.iter().enumerate() {
        let mut weight = one;
        for (k, &p) in point.iter().enumerate() {
            let bit = (i >> (num_vars - 1 - k)) & 1;
            weight *= if bit == 1 { p } else { one - p };
        }
        reference += coeff * weight;
    }
    assert_eq!(eval, reference, "evaluate disagrees with naive MLE reference");
});
