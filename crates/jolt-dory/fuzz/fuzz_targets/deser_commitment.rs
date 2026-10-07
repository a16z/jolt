#![no_main]

//! Decoding Dory commitments and openings from attacker bytes must never
//! panic.
//!
//! The input is tried as a serde (JSON, bincode) commitment, as a commitment
//! transcript atom, and as an argument string holding a commitment followed
//! by a transparent or ZK opening at a fixed statement. The last path drives
//! `receive_commitment`, the verifier's proof rebuild from the transcript,
//! and, for well-formed elements, `dory::verify`; every outcome must be `Ok`
//! or a typed error, and `finish` must then terminate without panicking.

use std::sync::OnceLock;

use jolt_dory::{DoryCommitment, DoryScheme, DoryVerifierSetup};
use jolt_field::{CanonicalDecode, Fr, Ring};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{Blake2b512, ProtocolId, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

/// Matches `tests/generate_seeds.rs`, so its seed openings parse in full.
const NUM_VARS: usize = 2;
const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-dory-fuzz/deser");
const SESSION: &[u8] = b"fuzz-deser";

fn verifier_setup() -> &'static DoryVerifierSetup {
    static SETUP: OnceLock<DoryVerifierSetup> = OnceLock::new();
    SETUP.get_or_init(|| DoryScheme::setup_verifier(NUM_VARS))
}

fuzz_target!(|data: &[u8]| {
    let config = bincode::config::standard();
    let _ = serde_json::from_slice::<DoryCommitment>(data);
    let _ = bincode::serde::decode_from_slice::<DoryCommitment, _>(data, config);
    let _ = DoryCommitment::from_bytes_le_checked(data);

    let setup = verifier_setup();
    let point: Vec<Fr> = (1..=NUM_VARS as u64).map(Fr::from_u64).collect();

    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, data);
    if let Ok(commitment) = DoryScheme::receive_commitment(setup, &mut transcript) {
        let _ = DoryScheme::verify(&commitment, &point, Fr::from_u64(0), setup, &mut transcript);
    }
    let _ = transcript.finish();

    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, data);
    if let Ok(commitment) = DoryScheme::receive_commitment(setup, &mut transcript) {
        let _ = DoryScheme::verify_zk(&commitment, &point, setup, &mut transcript);
    }
    let _ = transcript.finish();
});
