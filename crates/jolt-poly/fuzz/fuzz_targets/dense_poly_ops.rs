#![no_main]
use jolt_field::{Field, Fr};
use jolt_poly::Polynomial;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 32 bytes for a single field element to build a polynomial,
    // plus 32 bytes per point coordinate. Keep num_vars small to avoid OOM.
    if data.len() < 64 {
        return;
    }

    // Derive num_vars from first byte, capped at 8 (256 evaluations)
    let num_vars = (data[0] as usize % 8).max(1);
    let n = 1usize << num_vars;
    let needed = n * 32 + num_vars * 32;
    if data.len() < needed {
        return;
    }

    // Build evaluation vector from fuzzer data
    let evals: Vec<Fr> = (0..n)
        .map(|i| <Fr as Field>::from_bytes(&data[i * 32..(i + 1) * 32]))
        .collect();
    let poly = Polynomial::new(evals);

    // Build evaluation point from fuzzer data
    let point_start = n * 32;
    let point: Vec<Fr> = (0..num_vars)
        .map(|i| <Fr as Field>::from_bytes(&data[point_start + i * 32..point_start + (i + 1) * 32]))
        .collect();

    // evaluate and evaluate_and_consume must agree and not panic
    let eval = poly.evaluate(&point);
    let eval_consumed = poly.clone().evaluate_and_consume(&point);
    assert_eq!(eval, eval_consumed, "evaluate and evaluate_and_consume disagree");
});
