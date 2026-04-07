#![no_main]
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, PoseidonTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Blake2b: append arbitrary bytes + squeeze challenge — must never panic
    let mut blake = Blake2bTranscript::default();
    blake.append_bytes(data);
    let _ = blake.challenge();
    blake.append_bytes(data);
    let _ = blake.challenge();

    // Keccak: same exercise
    let mut keccak = KeccakTranscript::default();
    keccak.append_bytes(data);
    let _ = keccak.challenge();
    keccak.append_bytes(data);
    let _ = keccak.challenge();

    // Poseidon: most complex impl — field arithmetic + multi-chunk chaining
    let mut poseidon = PoseidonTranscript::default();
    poseidon.append_bytes(data);
    let _ = poseidon.challenge();
    poseidon.append_bytes(data);
    let _ = poseidon.challenge();
});
