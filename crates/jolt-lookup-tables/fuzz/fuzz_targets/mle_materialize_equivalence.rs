#![no_main]

//! Soundness oracle for every lookup table: the multilinear extension
//! (`evaluate_mle`) evaluated at a boolean point must equal the materialized
//! table entry (`materialize_entry`) at the corresponding index.
//!
//! This is the property the instruction sumcheck relies on — the prover
//! evaluates `evaluate_mle` at a random point, and its soundness rests on the
//! MLE being the correct extension of the materialized table. A disagreement
//! at any boolean point means the committed table is not the table the
//! verifier thinks it is.
//!
//! Runs at test size `XLEN = 8`, where the interleaved index is 16 bits and a
//! fuzzed `u16` covers the entire index space of all ~40 tables.

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_lookup_tables::tables::LookupTableKind;
use libfuzzer_sys::fuzz_target;

const XLEN: usize = 8;
const TABLE_BITS: usize = 2 * XLEN;

/// MSB-first boolean bit vector of `index` over `TABLE_BITS` bits, matching
/// the order `evaluate_mle` reads (`r[0]` is the most significant bit).
fn boolean_point(index: u128) -> Vec<Fr> {
    (0..TABLE_BITS)
        .rev()
        .map(|bit| Fr::from_u64(((index >> bit) & 1) as u64))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let kind_selector = data[0] as usize;
    let index = u16::from_le_bytes([data[1], data[2]]) as u128;
    let point = boolean_point(index);

    let count = LookupTableKind::<XLEN>::COUNT;
    let table = LookupTableKind::<XLEN>::iter()
        .nth(kind_selector % count)
        .expect("selector is reduced modulo the variant count");

    let materialized = Fr::from_u64(table.materialize_entry(index));
    let mle = table.evaluate_mle::<Fr, Fr>(&point);
    assert_eq!(
        materialized, mle,
        "MLE disagrees with materialized entry for {table:?} at index {index}"
    );
});
