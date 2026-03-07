#![no_main]
use jolt_field::{Field, FieldAccumulator, Fr, WideAccumulator};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least two pairs (128 bytes) so each half gets at least one.
    if data.len() < 128 {
        return;
    }

    let pairs = data.len() / 64;
    let split = pairs / 2;

    let mut acc1 = WideAccumulator::default();
    let mut acc2 = WideAccumulator::default();
    let mut acc_all = WideAccumulator::default();

    for i in 0..pairs {
        let offset = i * 64;
        let a = <Fr as Field>::from_bytes(&data[offset..offset + 32]);
        let b = <Fr as Field>::from_bytes(&data[offset + 32..offset + 64]);

        if i < split {
            acc1.fmadd(a, b);
        } else {
            acc2.fmadd(a, b);
        }
        acc_all.fmadd(a, b);
    }

    acc1.merge(acc2);

    assert_eq!(
        acc1.reduce(),
        acc_all.reduce(),
        "merge+reduce diverged from single-accumulator reduce ({pairs} pairs, split at {split})"
    );
});
