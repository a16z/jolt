#![no_main]

//! Argument-string mutation of accepted transparent proofs.
//!
//! Each input selects an accepted fixture (muldiv, advice consumer with both
//! advice kinds, committed-program muldiv) and applies exactly one mutation to
//! its NARG argument string or to the public statement the verifier binds;
//! see `jolt_verifier_fuzz::tamper_must_reject` for the input layout and the
//! mutation families. The verifier absorbs every argument-string byte and
//! decodes only canonical encodings, so every mutation that changes the
//! verifier's input must be rejected.

use std::sync::OnceLock;

use jolt_verifier_fuzz::{tamper_must_reject, Bundle};
use libfuzzer_sys::fuzz_target;

static FIXTURES: [&[u8]; 3] = [
    include_bytes!("../fixtures/muldiv-bundle.bin"),
    include_bytes!("../fixtures/advice-consumer-bundle.bin"),
    include_bytes!("../fixtures/committed-muldiv-bundle.bin"),
];

fn bundles() -> &'static [Bundle] {
    static BUNDLES: OnceLock<Vec<Bundle>> = OnceLock::new();
    BUNDLES.get_or_init(|| {
        FIXTURES
            .iter()
            .map(|bytes| Bundle::decode_verified(bytes))
            .collect()
    })
}

fuzz_target!(|data: &[u8]| {
    tamper_must_reject(bundles(), data, "transparent");
});
