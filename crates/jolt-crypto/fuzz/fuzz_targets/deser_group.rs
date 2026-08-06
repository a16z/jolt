#![no_main]

//! Group-element deserialization: no panics on arbitrary bytes, and every
//! accepted decode re-encodes canonically (decode → encode → decode is
//! stable and reproduces the same encoding).

use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, PedersenSetup};
use libfuzzer_sys::fuzz_target;

/// Accepted bincode decodes must re-encode to a fixed point: two encode
/// passes over the decoded value must agree byte-for-byte. Equality is
/// checked on the encodings so types without `PartialEq` participate too.
macro_rules! check_bincode {
    ($ty:ty, $data:expr, $config:expr) => {
        if let Ok((value, _)) = bincode::serde::decode_from_slice::<$ty, _>($data, $config) {
            let encoded =
                bincode::serde::encode_to_vec(&value, $config).expect("accepted value re-encodes");
            let (decoded, used) = bincode::serde::decode_from_slice::<$ty, _>(&encoded, $config)
                .expect("canonical encoding must decode");
            assert_eq!(used, encoded.len(), "canonical encoding has trailing bytes");
            let re_encoded = bincode::serde::encode_to_vec(&decoded, $config)
                .expect("decoded value re-encodes");
            assert_eq!(re_encoded, encoded, "decode → encode is not canonical");
        }
    };
}

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();

    let _ = serde_json::from_slice::<Bn254G1>(data);
    let _ = serde_json::from_slice::<Bn254G2>(data);
    let _ = serde_json::from_slice::<Bn254GT>(data);
    let _ = serde_json::from_slice::<PedersenSetup<Bn254G1>>(data);

    check_bincode!(Bn254G1, data, config);
    check_bincode!(Bn254G2, data, config);
    check_bincode!(Bn254GT, data, config);
    check_bincode!(PedersenSetup<Bn254G1>, data, config);
});
