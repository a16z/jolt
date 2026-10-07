#![no_main]

//! ZK/transparent mode separation for Dory openings.
//!
//! For a fuzzer-controlled polynomial and point: both modes must be complete
//! (an honest opening's argument string verifies in its own mode and is
//! consumed exactly), and each mode's verifier must reject the other mode's
//! argument string under the same protocol id and session. A ZK opening
//! carries `y_com`, Σ-protocol messages, and no final scalar-product message,
//! so accepting it transparently (or vice versa) would confuse two different
//! soundness contracts.

use std::sync::OnceLock;

use jolt_dory::{DoryCommitment, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use jolt_field::{CanonicalEncoding, Fr};
use jolt_openings::{CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const NUM_VARS: usize = 4;
const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-dory-fuzz/zk-mode");
const SESSION: &[u8] = b"fuzz-zk-mode";

fn setups() -> &'static (DoryProverSetup, DoryVerifierSetup) {
    static SETUPS: OnceLock<(DoryProverSetup, DoryVerifierSetup)> = OnceLock::new();
    SETUPS.get_or_init(|| {
        let prover_setup = DoryScheme::setup_prover(NUM_VARS);
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        (prover_setup, verifier_setup)
    })
}

fn verify(
    commitment: &DoryCommitment,
    point: &[Fr],
    eval: Fr,
    narg: &[u8],
) -> Result<(), OpeningsError> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    DoryScheme::verify(commitment, point, eval, &setups().1, &mut transcript)?;
    Ok(transcript.finish()?)
}

fn verify_zk(commitment: &DoryCommitment, point: &[Fr], narg: &[u8]) -> Result<(), OpeningsError> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let _y_com = DoryScheme::verify_zk(commitment, point, &setups().1, &mut transcript)?;
    Ok(transcript.finish()?)
}

fuzz_target!(|data: &[u8]| {
    let n = 1usize << NUM_VARS;
    if data.len() < (n + NUM_VARS) * SCALAR_BYTES {
        return;
    }
    let scalar_at = |index: usize| {
        let start = index * SCALAR_BYTES;
        <Fr as CanonicalEncoding>::from_bytes_le_reduced(&data[start..start + SCALAR_BYTES])
    };
    let evals: Vec<Fr> = (0..n).map(scalar_at).collect();
    let point: Vec<Fr> = (0..NUM_VARS).map(|i| scalar_at(n + i)).collect();
    let poly = Polynomial::new(evals);
    let eval = poly.evaluate(&point);

    let prover_setup = &setups().0;

    // ZK completeness.
    let (zk_commitment, zk_hint) =
        DoryScheme::commit_zk(poly.evaluations(), prover_setup).expect("commit_zk");
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let (_y_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, prover_setup, zk_hint, &mut transcript)
            .expect("open_zk");
    let zk_narg = transcript.finish();
    verify_zk(&zk_commitment, &point, &zk_narg).expect("honest ZK opening must verify in ZK mode");

    // Transparent completeness.
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), prover_setup).expect("commit");
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    DoryScheme::open(
        &poly,
        &point,
        eval,
        prover_setup,
        Some(hint),
        &mut transcript,
    )
    .expect("open");
    let narg = transcript.finish();
    verify(&commitment, &point, eval, &narg).expect("honest transparent opening must verify");

    // Mode confusion must be rejected in both directions.
    assert!(
        verify(&zk_commitment, &point, eval, &zk_narg).is_err(),
        "transparent verifier accepted a ZK opening"
    );
    assert!(
        verify_zk(&commitment, &point, &narg).is_err(),
        "ZK verifier accepted a transparent opening"
    );
});
