#![no_main]
use jolt_field::{Field, Fr};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }
    let a = <Fr as Field>::from_bytes(&data[..32]);
    let b = <Fr as Field>::from_bytes(&data[32..64]);

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

    // inverse must not panic
    if !a.is_zero() {
        let inv = a.inverse();
        assert_eq!(a * inv, Fr::from_u64(1));
    }

    // Prevent optimizing away
    let _ = (prod, sq);
});
