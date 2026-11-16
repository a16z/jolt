#![no_main]

//! Attacker-controlled bytes fed to the public bincode deserializer for
//! `JoltProof<AkitaScheme, _>`, and for the Akita commitment, opening proof
//! and verifier setup it or the verifier preprocessing carries, must never
//! panic or over-allocate — only decode successfully or return an error.
//! Decoding the three Akita types on their own reaches their byte payloads
//! without a valid proof prefix.

use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AkitaScheme, AkitaVerifierSetup};
use jolt_crypto::NoVectorCommitment;
use jolt_verifier::JoltProof;
use libfuzzer_sys::fuzz_target;

type FuzzProof = JoltProof<AkitaScheme, NoVectorCommitment<AkitaField>>;

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();
    // libFuzzer's malloc limit fails an allocation sized by an attacker-chosen
    // length prefix.
    let _ = bincode::serde::decode_from_slice::<FuzzProof, _>(data, config);
    let _ = bincode::serde::decode_from_slice::<AkitaCommitment, _>(data, config);
    let _ = bincode::serde::decode_from_slice::<AkitaBatchProof, _>(data, config);
    let _ = bincode::serde::decode_from_slice::<AkitaVerifierSetup, _>(data, config);
});
