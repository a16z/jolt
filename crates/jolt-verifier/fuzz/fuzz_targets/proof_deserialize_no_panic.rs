#![no_main]

//! Attacker-controlled proof bytes fed to the public bincode deserializer for
//! `JoltProof<DoryScheme, Pedersen>` must never panic or over-allocate — only
//! decode successfully or return an error. This is the genuine untrusted-input
//! boundary: a verifier receives serialized proofs from untrusted provers.

use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_verifier::JoltProof;
use libfuzzer_sys::fuzz_target;

type FuzzProof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();
    // The decoder must handle arbitrary bytes without panicking. A successful
    // decode is fine; libFuzzer's RSS limit guards against unbounded growth
    // from an attacker-chosen length prefix.
    let _ = bincode::serde::decode_from_slice::<FuzzProof, _>(data, config);
});
