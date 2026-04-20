#![no_main]

//! Fuzz: tamper with proof evaluation bytes and verify that verification rejects.
//!
//! We generate a valid proof, then corrupt an evaluation entry using fuzzer-chosen bytes.

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
    if data.len() < 10 {
        return;
    }

    let num_vars = 3;
    let n = 1usize << num_vars;

    let mut rng = ChaCha20Rng::seed_from_u64(0xfade);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = TestScheme::setup(&mut rng, n, g1, g2);
    let vk = TestScheme::verifier_setup(&pk);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);

    let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

    let mut pt = Blake2bTranscript::new(b"fuzz-tamper");
    let proof =
        <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt);

    let tamper_row = (data[0] as usize) % proof.v.len();
    let tamper_col = (data[1] as usize) % proof.v[tamper_row].len();
    let tamper_val = Fr::from_bytes(&data[2..]);

    // Skip if the corruption is a no-op
    if tamper_val == proof.v[tamper_row][tamper_col] {
        return;
    }

    let mut tampered = proof.clone();
    tampered.v[tamper_row][tamper_col] = tamper_val;

    let mut vt = Blake2bTranscript::new(b"fuzz-tamper");
    let result = <TestScheme as CommitmentScheme>::verify(
        &commitment,
        &point,
        eval,
        &tampered,
        &vk,
        &mut vt,
    );
    assert!(result.is_err(), "tampered proof must be rejected");
});
