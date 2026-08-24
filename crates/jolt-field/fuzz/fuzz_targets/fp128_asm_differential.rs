#![no_main]

use libfuzzer_sys::fuzz_target;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7};

fuzz_target!(|data: [u8; 96]| {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let a = Prime128OffsetA7F7::from_bytes_le_reduced(&data[..32]);
        let b = Prime128OffsetA7F7::from_bytes_le_reduced(&data[32..64]);
        let addend = Prime128OffsetA7F7::from_bytes_le_reduced(&data[64..96]);

        a.assert_asm_matches_portable_for_fuzzing(b, addend);
        a.assert_asm_matches_portable_for_fuzzing(a, addend);
        b.assert_asm_matches_portable_for_fuzzing(b, addend);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = data;
});
