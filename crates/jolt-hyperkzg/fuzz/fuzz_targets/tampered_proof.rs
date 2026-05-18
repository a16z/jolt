#![no_main]

//! Fuzz: tamper with proof fields and verify that verification rejects.
//!
//! We generate a valid proof, then corrupt either an evaluation entry,
//! an intermediate commitment, or a witness commitment using fuzzer-chosen bytes.

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{CommitmentScheme, CommitmentSchemeVerifier};
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
    let vk = TestScheme::prover_to_verifier_setup(&pk);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);

    let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

    let mut pt = Blake2bTranscript::new(b"fuzz-tamper");
    let proof =
        <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt);

    let mut tampered = proof.clone();

    match data[0] % 3 {
        0 => {
            // Tamper evaluation entries (exercises folding consistency checks)
            let tamper_row = (data[1] as usize) % tampered.v.len();
            let tamper_col = (data[2] as usize) % tampered.v[tamper_row].len();
            let tamper_val = Fr::from_bytes(&data[3..]);
            if tamper_val == proof.v[tamper_row][tamper_col] {
                return;
            }
            tampered.v[tamper_row][tamper_col] = tamper_val;
        }
        1 => {
            // Tamper intermediate commitments (exercises pairing check)
            if tampered.com.is_empty() {
                return;
            }
            let idx = (data[1] as usize) % tampered.com.len();
            let scalar = Fr::from_bytes(&data[2..]);
            tampered.com[idx] = tampered.com[idx].scalar_mul(&scalar);
            if tampered.com[idx] == proof.com[idx] {
                return;
            }
        }
        _ => {
            // Tamper witness commitments (exercises pairing check)
            let idx = (data[1] as usize) % tampered.w.len();
            let scalar = Fr::from_bytes(&data[2..]);
            tampered.w[idx] = tampered.w[idx].scalar_mul(&scalar);
            if tampered.w[idx] == proof.w[idx] {
                return;
            }
        }
    }

    let mut vt = Blake2bTranscript::new(b"fuzz-tamper");
    let result = <TestScheme as CommitmentSchemeVerifier>::verify(
        &commitment,
        &point,
        eval,
        &tampered,
        &vk,
        &mut vt,
    );
    assert!(result.is_err(), "tampered proof must be rejected");
});
