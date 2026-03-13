#![no_main]
use jolt_dory::{DoryCommitment, DoryProof};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Deserialization of arbitrary bytes must return Err, never panic.
    let _ = serde_json::from_slice::<DoryCommitment>(data);
    let _ = bincode::deserialize::<DoryCommitment>(data);
    let _ = serde_json::from_slice::<DoryProof>(data);
    let _ = bincode::deserialize::<DoryProof>(data);
});
