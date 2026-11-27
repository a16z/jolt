#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]
#![expect(
    unexpected_cfgs,
    reason = "the derive emits an allocative gate for consumers that define the feature"
)]

use jolt_claims::protocols::jolt::geometry::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS};
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_field::{JoltField, One, Zero, F128};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    CommittedRound, CommittedSumcheckProof, CompressedSumcheckProof, SumcheckProof,
};
use jolt_transcript::{protocol_tag, verifier_transcript, Blake2b512, ScriptedTranscript};
use jolt_verifier as Storage;
use jolt_verifier::stages::stage5::{RegistersValEvaluation, RegistersValEvaluationOutputClaims};
use jolt_verifier::VerifierError;
use jolt_verifier_derive::SumcheckBatch;

#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "Storage")]
struct MixedLengthSumchecks<F: JoltField> {
    long: RegistersValEvaluation<F>,
    absent: Option<RegistersValEvaluation<F>>,
    short: Option<RegistersValEvaluation<F>>,
}

fn batch() -> MixedLengthSumchecks<F128> {
    MixedLengthSumchecks {
        long: RegistersValEvaluation::new(TraceDimensions::new(2)),
        absent: None,
        short: Some(RegistersValEvaluation::new(TraceDimensions::new(1))),
    }
}

#[test]
#[expect(
    non_local_definitions,
    reason = "the derive exports a member-list macro even for a function-local batch"
)]
fn batch_can_be_declared_inside_a_function() {
    #[derive(SumcheckBatch)]
    struct LocalSumchecks<F: JoltField> {
        long: RegistersValEvaluation<F>,
        absent: Option<RegistersValEvaluation<F>>,
        short: Option<RegistersValEvaluation<F>>,
    }

    let batch = LocalSumchecks::<F128> {
        long: RegistersValEvaluation::new(TraceDimensions::new(2)),
        absent: None,
        short: Some(RegistersValEvaluation::new(TraceDimensions::new(1))),
    };
    let inputs = LocalInputPoints {
        long: RegistersValEvaluationInputClaims {
            registers_val: vec![F128::zero(); REGISTER_ADDRESS_BITS + 2],
        },
        absent: None,
        short: Some(RegistersValEvaluationInputClaims {
            registers_val: vec![F128::zero(); REGISTER_ADDRESS_BITS + 1],
        }),
    };
    let points = batch
        .derive_opening_points(&[F128::one(); 2], &inputs)
        .unwrap();
    assert!(points.short.is_some());
}

#[test]
fn crate_override_resolves_an_import_alias() {
    assert!(batch().empty_input_points().short.is_some());
}

#[test]
fn clear_verifies_mixed_lengths_with_zero_extension() {
    let batch = batch();
    let one = F128::one();
    let zero = F128::zero();
    let long_coefficient = F128::from_raw(2);
    let short_coefficient = F128::from_raw(3);
    let first_challenge = F128::from_raw(4);
    let second_challenge = F128::from_raw(8);
    let mut transcript = ScriptedTranscript::with_script(
        verifier_transcript::<Blake2b512>(&protocol_tag(b"batch-padding-clear"), b"", &[]),
        vec![
            long_coefficient,
            short_coefficient,
            first_challenge,
            second_challenge,
        ],
    );
    let challenges = batch.draw_challenges(&mut transcript).unwrap();

    let input = RegistersValEvaluationInputClaims { registers_val: one };
    let inputs = MixedLengthInputClaims {
        long: input.clone(),
        absent: None,
        short: Some(input),
    };
    let input_point = |rounds| {
        let mut registers_val = vec![zero; REGISTER_ADDRESS_BITS + rounds - 1];
        registers_val.push(one);
        RegistersValEvaluationInputClaims { registers_val }
    };
    let input_points = MixedLengthInputPoints {
        long: input_point(2),
        absent: None,
        short: Some(input_point(1)),
    };
    let openings = RegistersValEvaluationOutputClaims {
        rd_inc: one,
        rd_wa: one,
    };
    let output_claims = MixedLengthOutputClaims {
        long: openings.clone(),
        absent: None,
        short: Some(openings),
    };

    // With constant-one openings and a fixed cycle of 1, the long member is
    // (1 - x)(1 - y) and the short member is 1 - y. Its zero extension gives
    // the same polynomial as the long member.
    let combined = long_coefficient + short_coefficient;
    let second_sum = combined * (one - first_challenge);
    let proof = SumcheckProof::<F128, u64>::from(CompressedSumcheckProof {
        round_polynomials: vec![
            UnivariatePoly::new(vec![combined, -combined]).compress(),
            UnivariatePoly::new(vec![second_sum, -second_sum]).compress(),
        ],
        pow_nonces: Vec::new(),
    });

    let _ = batch
        .verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &output_claims,
            &proof,
            &mut transcript,
            1,
        )
        .unwrap();
}

#[test]
fn committed_rejects_zero_extension_before_touching_transcript() {
    let batch = batch();
    let mut transcript = ScriptedTranscript::new(verifier_transcript::<Blake2b512>(
        &protocol_tag(b"batch-padding-committed"),
        b"",
        &[],
    ));
    let proof = SumcheckProof::Committed(CommittedSumcheckProof {
        rounds: vec![
            CommittedRound {
                commitment: 1u64,
                degree: 1,
            },
            CommittedRound {
                commitment: 2u64,
                degree: 1,
            },
        ],
        output_claims: Default::default(),
    });

    assert!(matches!(
        batch.verify_zk(&proof, &mut transcript),
        Err(VerifierError::ProtocolAxisUnimplemented { .. })
    ));
    assert_eq!(transcript.events(), []);
}
