#![no_main]

use jolt_field::{
    FpExt4, FromPrimitiveInt, Invertible, Prime128Offset275, Prime31Offset19, ReducingBytes,
};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }

    let a31 = Prime31Offset19::from_le_bytes_mod_order(&data[..16]);
    let b31 = Prime31Offset19::from_le_bytes_mod_order(&data[16..32]);
    assert_eq!((a31 + b31) - b31, a31);
    assert_eq!((a31 - b31) + b31, a31);
    if !a31.is_zero() {
        assert_eq!(a31 * a31.inverse().unwrap(), Prime31Offset19::from_u64(1));
    }

    let a128 = Prime128Offset275::from_le_bytes_mod_order(&data[..32]);
    let b128 = Prime128Offset275::from_le_bytes_mod_order(&data[32..64]);
    assert_eq!((a128 + b128) - b128, a128);
    assert_eq!((a128 - b128) + b128, a128);
    if !a128.is_zero() {
        assert_eq!(
            a128 * a128.inverse().unwrap(),
            Prime128Offset275::from_u64(1)
        );
    }

    let extension = FpExt4::new([a31, b31, a31 + b31, a31 - b31]);
    if !extension.is_zero() {
        assert_eq!(
            extension * extension.inverse().unwrap(),
            FpExt4::<Prime31Offset19>::from_u64(1)
        );
    }
});
