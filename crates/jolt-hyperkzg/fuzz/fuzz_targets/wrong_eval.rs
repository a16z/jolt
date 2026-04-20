#![no_main]

//! Fuzz: claim a wrong evaluation and verify that verification rejects.
//!
//! The prover generates a valid proof for the correct evaluation. The
//! verifier checks against a fuzzer-derived wrong evaluation. Must reject.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestScheme = HyperKZGScheme<Bn254>;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let num_vars = 3;
    let n = 1usize << num_vars;

    let mut rng = ChaCha20Rng::seed_from_u64(0xface);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = TestScheme::setup(&mut rng, n, g1, g2);
    let vk = TestScheme::verifier_setup(&pk);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);

    let wrong_eval = Fr::from_bytes(data);
    if wrong_eval == eval {
        return;
    }

    let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

    let mut pt = Blake2bTranscript::new(b"fuzz-wrong-eval");
    let proof =
        <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt);

    let mut vt = Blake2bTranscript::new(b"fuzz-wrong-eval");
    let result = <TestScheme as CommitmentScheme>::verify(
        &commitment,
        &point,
        wrong_eval,
        &proof,
        &vk,
        &mut vt,
    );
    assert!(result.is_err(), "wrong evaluation must be rejected");
});
