#![no_main]
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Blake2b: append arbitrary bytes + squeeze challenge — must never panic
    let mut blake = Blake2bTranscript::default();
    blake.append_bytes(data);
    let _: jolt_field::Fr = blake.challenge();
    blake.append_bytes(data);
    let _: jolt_field::Fr = blake.challenge();

    // Keccak: same exercise
    let mut keccak = KeccakTranscript::default();
    keccak.append_bytes(data);
    let _: jolt_field::Fr = keccak.challenge();
    keccak.append_bytes(data);
    let _: jolt_field::Fr = keccak.challenge();
});
