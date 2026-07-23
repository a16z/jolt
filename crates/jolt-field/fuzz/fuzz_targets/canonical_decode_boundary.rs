#![no_main]

//! Boundary-biased canonical decoding: 32-byte encodings constructed around
//! the BN254 modulus (`< r`, `= r`, `r + small`, `2^256 − 1`, raw) must
//! decode to the `num-bigint` reference reduction and re-encode canonically.
//!
//! Complements `from_bytes`, which covers arbitrary-length inputs: random
//! bytes essentially never land within `[r, 2·r)`, so the wraparound branch
//! of the reduction needs deliberate biasing.

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

fn to_le_32(value: &BigUint) -> [u8; 32] {
    let mut out = [0u8; 32];
    let bytes = value.to_bytes_le();
    out[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 34 {
        return;
    }
    let class = data[0];
    let delta = u64::from_le_bytes(data[1..9].try_into().unwrap());
    let raw: [u8; 32] = data[2..34].try_into().unwrap();

    let candidate: [u8; 32] = match class % 5 {
        // Just below the modulus: r − 1 − (small delta).
        0 => to_le_32(&(modulus() - 1u8 - (BigUint::from(delta) % modulus()))),
        // Exactly the modulus (must reduce to zero).
        1 => to_le_32(modulus()),
        // Just above the modulus: the wraparound branch.
        2 => to_le_32(&(modulus() + BigUint::from(delta))),
        // All-ones ceiling.
        3 => [0xFF; 32],
        // Raw fuzzer bytes.
        _ => raw,
    };

    let decoded = <Fr as ReducingBytes>::from_le_bytes_mod_order(&candidate);
    let canonical = decoded.to_bytes_array();
    let got = BigUint::from_bytes_le(&canonical);

    assert!(&got < modulus(), "canonical encoding is not fully reduced");
    assert_eq!(
        got,
        BigUint::from_bytes_le(&candidate) % modulus(),
        "boundary decode disagrees with num-bigint reduction"
    );
});
