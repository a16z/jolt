#![no_main]
use jolt_field::{Field, FieldAccumulator, Fr, WideAccumulator};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

fuzz_target!(|data: &[u8]| {
    // Each pair of field elements needs 64 bytes (2 x 32-byte chunks).
    // Silently skip inputs that don't contain at least one complete pair.
    if data.len() < 64 {
        return;
    }

    let mut acc = WideAccumulator::default();
    let mut naive_sum = Fr::zero();

    let pairs = data.len() / 64;
    for i in 0..pairs {
        let offset = i * 64;
        let a = <Fr as Field>::from_bytes(&data[offset..offset + 32]);
        let b = <Fr as Field>::from_bytes(&data[offset + 32..offset + 64]);

        acc.fmadd(a, b);
        naive_sum += a * b;
    }

    assert_eq!(
        acc.reduce(),
        naive_sum,
        "WideAccumulator diverged from naive field arithmetic after {pairs} fmadd calls"
    );
});
