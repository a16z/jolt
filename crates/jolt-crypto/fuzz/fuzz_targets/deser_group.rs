#![no_main]
use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, PedersenSetup};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Deserialization of arbitrary bytes must return Err, never panic.
    let _ = serde_json::from_slice::<Bn254G1>(data);
    let _ = bincode::deserialize::<Bn254G1>(data);

    let _ = serde_json::from_slice::<Bn254G2>(data);
    let _ = bincode::deserialize::<Bn254G2>(data);

    let _ = serde_json::from_slice::<Bn254GT>(data);
    let _ = bincode::deserialize::<Bn254GT>(data);

    let _ = serde_json::from_slice::<PedersenSetup<Bn254G1>>(data);
    let _ = bincode::deserialize::<PedersenSetup<Bn254G1>>(data);
});
