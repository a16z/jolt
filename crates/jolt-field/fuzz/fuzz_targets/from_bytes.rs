#![no_main]

//! Differential check of `Fr::from_le_bytes_mod_order` against a `num-bigint`
//! reference reduction, plus canonicality of the re-encoding.

use std::sync::OnceLock;

use jolt_field::{FixedBytes, Fr, ReducingBytes};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigUint;

/// BN254 scalar-field modulus `r`.
fn modulus() -> &'static BigUint {
    static R: OnceLock<BigUint> = OnceLock::new();
    R.get_or_init(|| {
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .expect("BN254 r literal parses")
    })
}

fuzz_target!(|data: &[u8]| {
    let a = <Fr as ReducingBytes>::from_le_bytes_mod_order(data);
    let canonical = a.to_bytes_array();

    let got = BigUint::from_bytes_le(&canonical);
    assert!(&got < modulus(), "canonical encoding is not fully reduced");
    assert_eq!(
        got,
        BigUint::from_bytes_le(data) % modulus(),
        "from_le_bytes_mod_order disagrees with num-bigint reduction"
    );
});
