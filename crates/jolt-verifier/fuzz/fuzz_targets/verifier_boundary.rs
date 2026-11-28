#![no_main]

//! Byte-level mutation of accepted verifier bundles.
//!
//! A mutated bundle may fail canonical decoding or verification. If it passes
//! both gates, its bytes must exactly match the accepted fixture selected by
//! the fuzz input.

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::Fr;
use jolt_transcript::{Keccak, VerifierState};
use jolt_verifier::{
    deserialize_verifier_object, verify, JoltProof, JoltVerifierPreprocessing,
};
use libfuzzer_sys::fuzz_target;

type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type Bundle = (Preprocessing, JoltDevice, Proof, Option<DoryCommitment>);

const MAX_MUTATIONS: usize = 8;
const FIXTURES: [&[u8]; 3] = [
    include_bytes!("../fixtures/muldiv-bundle.bin"),
    include_bytes!("../fixtures/advice-consumer-bundle.bin"),
    include_bytes!("../fixtures/committed-muldiv-bundle.bin"),
];

fn position(data: &[u8], len: usize) -> usize {
    let mut bytes = [0u8; 8];
    let count = data.len().min(bytes.len());
    bytes[..count].copy_from_slice(&data[..count]);
    usize::from_le_bytes(bytes) % len.max(1)
}

fn mutate(candidate: &mut Vec<u8>, data: &[u8]) {
    for mutation in data.chunks(10).take(MAX_MUTATIONS) {
        let Some((&operation, arguments)) = mutation.split_first() else {
            return;
        };
        let index = position(arguments, candidate.len());
        let value = arguments.last().copied().unwrap_or(1);
        match operation % 6 {
            0 if !candidate.is_empty() => candidate[index] ^= value.max(1),
            1 if !candidate.is_empty() => candidate[index] = value,
            2 => candidate.insert(index, value),
            3 if !candidate.is_empty() => {
                candidate.remove(index);
            }
            4 if !candidate.is_empty() => candidate.truncate(index),
            _ => candidate.push(value),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&selector, mutations)) = data.split_first() else {
        return;
    };
    let fixture = FIXTURES[selector as usize % FIXTURES.len()];
    let mut candidate = fixture.to_vec();
    mutate(&mut candidate, mutations);

    let Ok((preprocessing, public_io, proof, advice)) =
        deserialize_verifier_object::<Bundle>(&candidate)
    else {
        return;
    };
    let result = verify::<Fr, DoryScheme, Pedersen<Bn254G1>, VerifierState<'_, Keccak>>(
        &preprocessing,
        &public_io,
        &proof,
        advice.as_ref(),
    );
    assert!(
        candidate == fixture || result.is_err(),
        "verifier accepted a non-byte-identical proof bundle"
    );
});
