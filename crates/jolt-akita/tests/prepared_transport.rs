//! A prepared verifier setup survives transport the way a guest receives it:
//! payloads detached from the record and attached back in place.
#![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]

use std::sync::Arc;

use jolt_akita::{
    AkitaField, AkitaNativeBatching, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams,
    AkitaVerifierSetup, TrustedBytes,
};
use jolt_field::Ring;
use jolt_openings::{BatchOpeningScheme, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

#[test]
fn detached_payloads_verify_once_attached_in_place() {
    let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
    let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
        14,
        1,
        [7; 32],
        Arc::clone(&artifacts),
    ))
    .unwrap();
    let polynomial = Polynomial::new(
        (0..(1u64 << 14))
            .map(|i| AkitaField::from_u64(3 + 7 * i))
            .collect(),
    );
    let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover_setup).unwrap();
    let point = (5..19).map(AkitaField::from_u64).collect::<Vec<_>>();
    let value = polynomial.evaluate(&point);
    let statement = vec![VerifierOpeningClaim {
        commitment,
        evaluation: EvaluationClaim::new(point, value),
    }];
    let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"prepared");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        vec![&polynomial],
        hint,
        &mut prover_transcript,
    )
    .unwrap();

    let mut prepared = verifier_setup;
    prepared
        .prepare_verifier(proof.schedule_row_digest())
        .unwrap();
    let config = bincode::config::standard();
    let encoded = bincode::serde::encode_to_vec(&prepared, config).unwrap();
    let (mut transported, _): (AkitaVerifierSetup, _) =
        bincode::serde::decode_from_slice(&encoded, config).unwrap();
    let bodies = transported.detach_prepared_payloads().unwrap();
    let bodies: Vec<TrustedBytes> = bodies
        .into_iter()
        // SAFETY: the bodies are this test's own detached payloads.
        .map(|body| unsafe { TrustedBytes::new(Box::leak(body.into_boxed_slice())) })
        .collect();

    assert!(transported
        .clone()
        .attach_prepared_payloads(&bodies[1..])
        .is_err());
    transported.attach_prepared_payloads(&bodies).unwrap();
    let mut transcript = Blake2bTranscript::<AkitaField>::new(b"prepared");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &transported,
        &statement,
        &proof,
        &mut transcript,
    )
    .unwrap();
}
