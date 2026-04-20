#![no_main]
use jolt_dory::{DoryCommitment, DoryProof};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();

    let _ = serde_json::from_slice::<DoryCommitment>(data);
    let _ = bincode::serde::decode_from_slice::<DoryCommitment, _>(data, config);
    let _ = serde_json::from_slice::<DoryProof>(data);
    let _ = bincode::serde::decode_from_slice::<DoryProof, _>(data, config);
});
