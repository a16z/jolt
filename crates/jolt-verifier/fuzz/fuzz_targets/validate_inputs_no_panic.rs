#![no_main]

//! The verifier's pre-crypto input checks — proof-header decoding, then
//! `validate_inputs`' memory-layout match, input/output size bounds, and
//! trace-length, RAM-size, one-hot, read-write, and trace-order validity —
//! over attacker-chosen header bytes and public I/O sizes. They must return
//! a typed `Ok`/`Err` for any input, never panic or over-allocate.
//!
//! The honest preprocessing and public I/O come from the checked-in muldiv
//! fixture. Input layout: `data[0]` bit 0 declares a trusted-advice
//! commitment, `data[1]` sets the public input length in 64-byte units, and
//! `data[2..]` is read as an argument string: the proof header first, with
//! the bytes after it becoming the public outputs.

use std::sync::OnceLock;

use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_transcript::VerifierTranscript;
use jolt_verifier::{jolt_protocol_id, validate_inputs, JoltSponge, ProofHeader, JOLT_SESSION};
use jolt_verifier_fuzz::Bundle;
use libfuzzer_sys::fuzz_target;

static FIXTURE: &[u8] = include_bytes!("../fixtures/muldiv-bundle.bin");

fn bundle() -> &'static Bundle {
    static BUNDLE: OnceLock<Bundle> = OnceLock::new();
    BUNDLE.get_or_init(|| Bundle::decode(FIXTURE))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let bundle = bundle();
    let trusted_advice_present = data[0] & 1 == 1;
    let narg = &data[2..];
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        narg,
    );
    let Ok(header) = ProofHeader::receive(&mut transcript) else {
        return;
    };
    let header_len = narg.len() - transcript.remaining();

    let mut public_io = bundle.public_io.clone();
    public_io.inputs = vec![0u8; data[1] as usize * 64];
    public_io.outputs = narg[header_len..].to_vec();

    let _ = validate_inputs::<DoryScheme, Pedersen<Bn254G1>>(
        &bundle.preprocessing,
        &public_io,
        &header,
        trusted_advice_present,
    );
});
