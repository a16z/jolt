#![no_main]

//! Completeness: an honest commit/open of a fuzzer-controlled polynomial at a
//! fuzzer-controlled point must always verify.
//!
//! Replaces the old `commit_open_verify` target, whose input collapsed to a
//! u64 RNG seed and which rebuilt the SRS every iteration. The SRS lives in a
//! `OnceLock`; the fuzzer controls every coefficient and point coordinate.

use std::sync::OnceLock;

use jolt_crypto::Bn254;
use jolt_field::{Fr, ReducingBytes};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestScheme = HyperKZGScheme<Bn254>;

/// Bytes per BN254 scalar window.
const SCALAR_BYTES: usize = 32;

/// Largest variable count whose full encoding fits the 4096-byte input cap.
const MAX_NUM_VARS: usize = 6;

fn srs() -> &'static (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    static SRS: OnceLock<(HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>)> =
        OnceLock::new();
    SRS.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xf00d);
        let pk = TestScheme::setup(
            &mut rng,
            1usize << MAX_NUM_VARS,
            Bn254::g1_generator(),
            Bn254::g2_generator(),
        );
        let vk = TestScheme::verifier_setup(&pk);
        (pk, vk)
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let num_vars = (data[0] as usize % MAX_NUM_VARS) + 1; // 1..=6, all reachable
    let n = 1usize << num_vars;
    if data.len() < 1 + (n + num_vars) * SCALAR_BYTES {
        return;
    }

    let scalar_at = |index: usize| {
        let start = 1 + index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let evals: Vec<Fr> = (0..n).map(scalar_at).collect();
    let point: Vec<Fr> = (0..num_vars).map(|i| scalar_at(n + i)).collect();

    let poly = Polynomial::new(evals);
    let eval = poly.evaluate(&point);

    let (pk, vk) = srs();
    let (commitment, ()) =
        TestScheme::commit(poly.evaluations(), pk).expect("commit of a valid polynomial");

    let mut pt = Blake2bTranscript::new(b"fuzz-completeness");
    let proof = <TestScheme as CommitmentScheme>::open(&poly, &point, eval, pk, None, &mut pt)
        .expect("open of a valid polynomial");

    let mut vt = Blake2bTranscript::new(b"fuzz-completeness");
    <TestScheme as CommitmentScheme>::verify(&commitment, &point, eval, &proof, vk, &mut vt)
        .expect("honest opening must verify");
});
