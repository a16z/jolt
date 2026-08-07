#![no_main]

//! ZK/transparent mode separation for Dory openings.
//!
//! For a fuzzer-controlled polynomial and point: both modes must be complete
//! (an honest opening verifies in its own mode), and each mode's verifier
//! must reject the other mode's proof — a ZK proof carries `y_com` and no
//! final scalar-product message, so accepting it transparently (or vice
//! versa) would confuse two different soundness contracts.

use std::sync::OnceLock;

use jolt_dory::{DoryProverSetup, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, ReducingBytes};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const NUM_VARS: usize = 4;
const TRANSCRIPT_LABEL: &[u8] = b"fuzz-zk-mode";

fn setups() -> &'static (DoryProverSetup, DoryVerifierSetup) {
    static SETUPS: OnceLock<(DoryProverSetup, DoryVerifierSetup)> = OnceLock::new();
    SETUPS.get_or_init(|| {
        (
            DoryScheme::setup_prover(NUM_VARS),
            DoryScheme::setup_verifier(NUM_VARS),
        )
    })
}

fuzz_target!(|data: &[u8]| {
    let n = 1usize << NUM_VARS;
    if data.len() < (n + NUM_VARS) * SCALAR_BYTES {
        return;
    }
    let scalar_at = |index: usize| {
        let start = index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let evals: Vec<Fr> = (0..n).map(scalar_at).collect();
    let point: Vec<Fr> = (0..NUM_VARS).map(|i| scalar_at(n + i)).collect();
    let poly = Polynomial::new(evals);
    let eval = poly.evaluate(&point);

    let (prover_setup, verifier_setup) = setups();

    // ZK completeness.
    let (zk_commitment, zk_hint) =
        DoryScheme::commit_zk(poly.evaluations(), prover_setup).expect("commit_zk");
    let mut pt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    let (zk_proof, _y_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, prover_setup, zk_hint, &mut pt)
            .expect("open_zk");
    let mut vt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    DoryScheme::verify_zk(&zk_commitment, &point, &zk_proof, verifier_setup, &mut vt)
        .expect("honest ZK opening must verify in ZK mode");

    // Transparent completeness.
    let (commitment, hint) =
        DoryScheme::commit(poly.evaluations(), prover_setup).expect("commit");
    let mut pt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    let proof = DoryScheme::open(&poly, &point, eval, prover_setup, Some(hint), &mut pt)
        .expect("open");
    let mut vt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    DoryScheme::verify(&commitment, &point, eval, &proof, verifier_setup, &mut vt)
        .expect("honest transparent opening must verify");

    // Mode confusion must be rejected in both directions.
    let mut vt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    assert!(
        DoryScheme::verify(
            &zk_commitment,
            &point,
            eval,
            &zk_proof,
            verifier_setup,
            &mut vt,
        )
        .is_err(),
        "transparent verifier accepted a ZK proof"
    );
    let mut vt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    assert!(
        DoryScheme::verify_zk(&commitment, &point, &proof, verifier_setup, &mut vt).is_err(),
        "ZK verifier accepted a transparent proof"
    );
});
