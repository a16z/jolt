#![no_main]
use jolt_field::{Fr, FromPrimitiveInt, Invertible, ReducingBytes};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }
    let a = <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[..32]);
    let b = <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[32..64]);

    // Arithmetic operations must not panic
    let sum = a + b;
    let diff = a - b;
    let prod = a * b;
    let sq = a * a;

    // (a + b) - b == a
    assert_eq!(sum - b, a);
    // (a - b) + b == a
    assert_eq!(diff + b, a);
    // a * 0 == 0
    assert!((a * Fr::zero()).is_zero());

    // a·(a+b) == a² + a·b ties the otherwise-unchecked product and square
    // into a distributivity identity.
    assert_eq!(a * sum, sq + prod, "distributivity violated");

    // inverse must not panic
    if !a.is_zero() {
        let inv = a.inverse().expect("nonzero element must have inverse");
        assert_eq!(a * inv, Fr::from_u64(1));
    }
});
