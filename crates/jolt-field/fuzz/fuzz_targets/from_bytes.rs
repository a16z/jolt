#![no_main]
use jolt_field::{Field, Fr};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // from_bytes should never panic on arbitrary input
    let a = <Fr as Field>::from_bytes(data);

    // Round-trip: from_bytes → to_bytes → from_bytes must be stable
    let bytes = a.to_bytes();
    let b = <Fr as Field>::from_bytes(&bytes);
    let bytes2 = b.to_bytes();
    assert_eq!(bytes, bytes2, "from_bytes round-trip is not stable");
});
