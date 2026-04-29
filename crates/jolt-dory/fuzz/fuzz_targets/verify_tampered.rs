#![no_main]

use std::sync::OnceLock;

use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::Blake2bTranscript;
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

struct Fixture {
    verifier_setup: DoryVerifierSetup,
    commitment: DoryCommitment,
    point: Vec<Fr>,
    eval: Fr,
}

fn fixture() -> &'static Fixture {
    static FIX: OnceLock<Fixture> = OnceLock::new();
    FIX.get_or_init(|| {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(0xF0_22);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let (commitment, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        Fixture {
            verifier_setup,
            commitment,
            point,
            eval,
        }
    })
}

fuzz_target!(|data: &[u8]| {
    let fix = fixture();

    let config = bincode::config::standard();
    let proof: DoryProof = match bincode::serde::decode_from_slice(data, config) {
        Ok((p, _)) => p,
        Err(_) => match serde_json::from_slice::<DoryProof>(data) {
            Ok(p) => p,
            Err(_) => return,
        },
    };

    let mut transcript = Blake2bTranscript::new(b"fuzz-tampered");
    let result = DoryScheme::verify(
        &fix.commitment,
        &fix.point,
        fix.eval,
        &proof,
        &fix.verifier_setup,
        &mut transcript,
    );
    assert!(
        result.is_err(),
        "verify accepted an attacker-controlled proof against a fixed commitment",
    );
});
