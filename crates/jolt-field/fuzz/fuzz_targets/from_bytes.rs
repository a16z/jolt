#![no_main]
use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // from_bytes should never panic on arbitrary input
    let a = <Fr as CanonicalEncoding>::from_bytes_le_reduced(data);

    // Round-trip: from_bytes → to_bytes → from_bytes must be stable
    let bytes = a.to_bytes_le_vec();
    let b = <Fr as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
    let bytes2 = b.to_bytes_le_vec();
    assert_eq!(bytes, bytes2, "from_bytes round-trip is not stable");
});
