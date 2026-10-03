#![no_main]

use std::sync::OnceLock;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_verifier::{validate_inputs_from_parts, JoltProof, JoltVerifierPreprocessing};
use libfuzzer_sys::fuzz_target;

type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type Bundle = (Preprocessing, JoltDevice, Proof, Option<DoryCommitment>);

static FIXTURE: &[u8] = include_bytes!("../fixtures/muldiv-bundle.bin");

fn bundle() -> &'static Bundle {
    static BUNDLE: OnceLock<Bundle> = OnceLock::new();
    BUNDLE.get_or_init(|| {
        let (bundle, _): (Bundle, usize) =
            bincode::serde::decode_from_slice(FIXTURE, bincode::config::standard())
                .expect("fixture decodes");
        bundle
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 18 {
        return;
    }
    let (preprocessing, public_io, proof, _) = bundle();

    let trace_length = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let ram_k = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let trusted_present = data[16] & 1 == 1;
    let untrusted_present = data[16] & 2 == 2;
    let zk = data[16] & 4 == 4;

    let input_len = (data[17] as usize) * 64;
    let mut io = public_io.clone();
    io.inputs = vec![0u8; input_len.min(1 << 16)];
    io.outputs = data[18..].to_vec();

    let _ = validate_inputs_from_parts::<DoryScheme, Pedersen<Bn254G1>>(
        preprocessing,
        &io,
        trace_length,
        ram_k,
        proof.trace_polynomial_order,
        proof.one_hot_config,
        trusted_present,
        untrusted_present,
        zk,
    );
});
