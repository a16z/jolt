//! Deep fold-schedule coverage. The rest of the suite stays at the
//! 13/14-variable planner floor where schedules carry one to three recursive
//! folds; this exercises a deeper recursion (16 variables, four recursive
//! folds) end to end, plus the `valid_proof || garbage` rejection the
//! adapter's exact-length deserializer must enforce.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaBatchProof, AkitaScheme};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{f, layout, polynomial, setup_for};

struct ProofFixture {
    verifier_setup: <AkitaScheme as CommitmentScheme>::VerifierSetup,
    commitment: jolt_akita::AkitaCommitment,
    point: Vec<jolt_akita::AkitaField>,
    eval: jolt_akita::AkitaField,
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

/// 16 variables resolve to four recursive fold levels — deeper than any
/// other suite fixture — and a tampered evaluation must still reject.
#[test]
fn deep_recursive_fold_schedule_roundtrips() {
    let fixture = fold_roundtrip(16, b"akita-fold-deep");

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

/// `valid_proof || garbage` must be rejected: the adapter's deserializer
/// requires backend payloads to consume their byte buffers exactly.
#[test]
fn proof_payloads_with_trailing_garbage_reject() {
    let fixture = fold_roundtrip(13, b"akita-fold-trailing");

    for field in ["serialized_akita_proof", "serialized_akita_proof_shape"] {
        let mut value =
            serde_json::to_value(&fixture.proof).expect("proof should serialize to JSON");
        value
            .get_mut(field)
            .expect("proof should expose the payload")
            .as_array_mut()
            .expect("payload should serialize as a byte array")
            .push(serde_json::json!(0));
        let extended: AkitaBatchProof =
            serde_json::from_value(value).expect("extended proof should deserialize");

        let err = fixture
            .verify(&extended)
            .expect_err("trailing payload bytes must be rejected");
        assert!(
            matches!(
                &err,
                OpeningsError::InvalidBatch(message) if message.contains("trailing bytes")
            ),
            "expected a trailing-bytes rejection for {field}, got: {err}"
        );
    }
}
