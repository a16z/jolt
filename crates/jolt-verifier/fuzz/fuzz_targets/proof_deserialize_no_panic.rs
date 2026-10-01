#![no_main]

//! Attacker-controlled proof bytes must never panic or over-allocate the
//! verifier's untrusted-input boundary, only decode or verify to a typed
//! result:
//!
//! - `data` as a serialized `JoltProof` through the public bincode decoder,
//!   verifying whatever decodes;
//! - `data` as the argument string of a proof with the honest protocol
//!   configuration, verified against the muldiv fixture. Unless it equals the
//!   honest argument string it must be rejected.

use std::sync::OnceLock;

use jolt_verifier::JoltProof;
use jolt_verifier_fuzz::Bundle;
use libfuzzer_sys::fuzz_target;

static FIXTURE: &[u8] = include_bytes!("../fixtures/muldiv-bundle.bin");

fn bundle() -> &'static Bundle {
    static BUNDLE: OnceLock<Bundle> = OnceLock::new();
    BUNDLE.get_or_init(|| Bundle::decode_verified(FIXTURE))
}

fuzz_target!(|data: &[u8]| {
    let bundle = bundle();
    // libFuzzer's RSS limit guards against unbounded growth from an
    // attacker-chosen length prefix.
    if let Ok((proof, _)) =
        bincode::serde::decode_from_slice::<JoltProof, _>(data, bincode::config::standard())
    {
        let _ = bundle.verify(&proof);
    }

    if data == bundle.proof.narg.as_slice() {
        return;
    }
    let proof = JoltProof {
        protocol: bundle.proof.protocol,
        narg: data.to_vec(),
    };
    assert!(
        bundle.verify(&proof).is_err(),
        "verifier accepted an arbitrary argument string"
    );
});
