#![no_main]
use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, PedersenSetup};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();

    let _ = serde_json::from_slice::<Bn254G1>(data);
    let _ = bincode::serde::decode_from_slice::<Bn254G1, _>(data, config);

    let _ = serde_json::from_slice::<Bn254G2>(data);
    let _ = bincode::serde::decode_from_slice::<Bn254G2, _>(data, config);

    let _ = serde_json::from_slice::<Bn254GT>(data);
    let _ = bincode::serde::decode_from_slice::<Bn254GT, _>(data, config);

    let _ = serde_json::from_slice::<PedersenSetup<Bn254G1>>(data);
    let _ = bincode::serde::decode_from_slice::<PedersenSetup<Bn254G1>, _>(data, config);
});
