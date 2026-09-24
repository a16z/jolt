#![no_main]

use jolt_field::{CanonicalEncoding, Prime64Offset59, Ring};
use libfuzzer_sys::fuzz_target;

const MODULUS: u128 = (u64::MAX as u128) - 58;

fuzz_target!(|data: [u8; 32]| {
    let mut a_bytes = [0u8; 16];
    let mut b_bytes = [0u8; 16];
    a_bytes.copy_from_slice(&data[..16]);
    b_bytes.copy_from_slice(&data[16..]);
    let a_raw = u128::from_le_bytes(a_bytes);
    let b_raw = u128::from_le_bytes(b_bytes);
    let a_value = a_raw % MODULUS;
    let b_value = b_raw % MODULUS;
    let a = Prime64Offset59::from_u128_reduced(a_raw);
    let b = Prime64Offset59::from_u128_reduced(b_raw);

    assert_eq!((a + b).to_u128_checked(), Some((a_value + b_value) % MODULUS));
    assert_eq!(
        (a - b).to_u128_checked(),
        Some((a_value + MODULUS - b_value) % MODULUS)
    );
    assert_eq!((a * b).to_u128_checked(), Some((a_value * b_value) % MODULUS));
    assert_eq!(a.square().to_u128_checked(), Some((a_value * a_value) % MODULUS));
});
