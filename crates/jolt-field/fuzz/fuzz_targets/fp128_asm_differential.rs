#![no_main]

use libfuzzer_sys::fuzz_target;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use jolt_field::{CanonicalEncoding, Fp128, Prime128Offset275, Prime128OffsetA7F7};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
type GenericOffset173 = Fp128<{ u128::MAX - 172 }>;

fuzz_target!(|data: [u8; 96]| {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        macro_rules! exercise {
            ($field:ty) => {{
                let a = <$field>::from_bytes_le_reduced(&data[..32]);
                let b = <$field>::from_bytes_le_reduced(&data[32..64]);
                let addend = <$field>::from_bytes_le_reduced(&data[64..96]);

                a.assert_asm_matches_portable_for_fuzzing(b, addend);
                a.assert_asm_matches_portable_for_fuzzing(a, addend);
                b.assert_asm_matches_portable_for_fuzzing(b, addend);
            }};
        }

        // A test-only offset outside the published field aliases ensures the
        // register-parameterized kernels cannot become unreachable unnoticed.
        exercise!(GenericOffset173);
        exercise!(Prime128Offset275);
        exercise!(Prime128OffsetA7F7);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = data;
});
