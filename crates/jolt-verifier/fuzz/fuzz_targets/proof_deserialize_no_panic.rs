#![no_main]

use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_verifier::JoltProof;
use libfuzzer_sys::fuzz_target;

type FuzzProof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();
    let _ = bincode::serde::decode_from_slice::<FuzzProof, _>(data, config);
});
