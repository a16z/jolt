#![no_main]

//! Differential check of the three eq-polynomial evaluation paths: the
//! materialized table (`EqPolynomial::evaluations`), the per-index formula
//! (`eq_index_msb`), and the split tensor table (`TensorEqTable`) must agree
//! on every hypercube index.

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::{eq_index_msb, EqPolynomial, TensorEqTable};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 10;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let num_vars = (data[0] as usize % MAX_NUM_VARS) + 1; // 1..=10
    if data.len() < 1 + num_vars * SCALAR_BYTES {
        return;
    }
    let point: Vec<Fr> = (0..num_vars)
        .map(|i| {
            let start = 1 + i * SCALAR_BYTES;
            <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
        })
        .collect();

    let table = EqPolynomial::new(point.clone()).evaluations();
    let tensor = TensorEqTable::new(&point);
    assert_eq!(table.len(), 1 << num_vars);

    for (index, &entry) in table.iter().enumerate() {
        assert_eq!(
            entry,
            eq_index_msb(&point, index as u128),
            "eq table disagrees with eq_index_msb at index {index}"
        );
        assert_eq!(
            entry,
            tensor.evaluate_index(index),
            "eq table disagrees with TensorEqTable at index {index}"
        );
    }
});
