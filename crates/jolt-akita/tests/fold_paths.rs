//! Deep fold-schedule coverage at 17 variables, with at least four recursive
//! folds, plus the `valid_proof || garbage` rejection Akita's argument parser
//! must enforce.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

#[expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]
mod support;

use akita_params::{PolynomialGroupLayout, ScheduleLookupKey};
use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme,
};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{f, layout, polynomial, setup_for};

struct ProofFixture {
    verifier_setup: <AkitaScheme as CommitmentScheme>::VerifierSetup,
    commitment: AkitaCommitment,
    point: Vec<AkitaField>,
    eval: AkitaField,
    proof: AkitaBatchProof,
    label: &'static [u8],
}

impl ProofFixture {
    fn verify(&self, proof: &AkitaBatchProof) -> Result<(), OpeningsError> {
        let mut transcript = Blake2bTranscript::new(self.label);
        AkitaScheme::verify(
            &self.commitment,
            &self.point,
            self.eval,
            proof,
            &self.verifier_setup,
            &mut transcript,
        )
    }
}

fn fold_roundtrip(num_vars: usize, label: &'static [u8]) -> ProofFixture {
    let (prover_setup, verifier_setup) = setup_for(num_vars, 1, layout(7));
    let poly = polynomial(num_vars, 5);
    let point: Vec<_> = (0..num_vars).map(|index| f(index as u64 + 2)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit(&poly, &prover_setup).expect("dense commit should succeed");

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof = AkitaScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .expect("fold-schedule proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(label);
    AkitaScheme::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("fold-schedule proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    ProofFixture {
        verifier_setup,
        commitment,
        point,
        eval,
        proof,
        label,
    }
}

/// A deep schedule must reject a tampered evaluation. Preserve the minimum
/// recursive depth across catalog regeneration without pinning the optimizer's
/// exact choice of fold count.
#[test]
fn deep_recursive_fold_schedule_roundtrips() {
    const NUM_VARS: usize = 17;
    let depth = AkitaScheduleArtifacts::shared_from_default_directory()
        .dense_catalog()
        .expect("dense catalog")
        .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
            NUM_VARS, 1,
        )))
        .expect("deep fixture row must resolve")
        .schedule()
        .recursive_folds
        .len();
    assert!(depth >= 4, "the fixture must exercise deep recursion");
    let fixture = fold_roundtrip(NUM_VARS, b"akita-fold-deep");

    let mut tampered_eval = fixture.eval;
    tampered_eval += f(1);
    let mut transcript = Blake2bTranscript::new(fixture.label);
    assert!(
        AkitaScheme::verify(
            &fixture.commitment,
            &fixture.point,
            tampered_eval,
            &fixture.proof,
            &fixture.verifier_setup,
            &mut transcript,
        )
        .is_err(),
        "tampered evaluation must reject on the deep fold path too"
    );
}

#[test]
fn proof_payloads_with_trailing_or_missing_bytes_reject() {
    let fixture = fold_roundtrip(14, b"akita-fold-trailing");

    let mut value = serde_json::to_value(&fixture.proof).expect("proof should serialize to JSON");
    value
        .get_mut("backend_proof")
        .expect("proof should expose the payload")
        .as_array_mut()
        .expect("payload should serialize as a byte array")
        .push(serde_json::json!(0));
    let extended: AkitaBatchProof =
        serde_json::from_value(value.clone()).expect("extended proof should deserialize");

    let err = fixture
        .verify(&extended)
        .expect_err("trailing payload bytes must be rejected");
    assert!(
        matches!(&err, OpeningsError::VerificationFailed),
        "expected a verification failure, got: {err}"
    );

    let proof_len = fixture.proof.backend_proof_body_size();
    for length in [0, 1, proof_len / 2, proof_len - 1] {
        let mut truncated = value.clone();
        truncated["backend_proof"]
            .as_array_mut()
            .expect("payload should serialize as a byte array")
            .truncate(length);
        let truncated: AkitaBatchProof =
            serde_json::from_value(truncated).expect("truncated proof should deserialize");
        assert!(
            matches!(
                fixture.verify(&truncated),
                Err(OpeningsError::VerificationFailed)
            ),
            "proof truncated to {length} bytes must reject"
        );
    }
}
