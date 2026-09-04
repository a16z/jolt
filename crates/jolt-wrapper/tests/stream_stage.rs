#![expect(
    clippy::expect_used,
    reason = "test fixtures fail immediately on protocol errors"
)]

use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake3Transcript, Transcript};
use jolt_wrapper::stream::{
    prove_stage, verify_stage_with, StageMember, StageMemberSpec, StreamError,
};

struct LinearSum {
    polynomial: Polynomial<Fr>,
    rounds: usize,
}

impl LinearSum {
    fn new(evaluations: Vec<Fr>) -> Self {
        let rounds = evaluations.len().trailing_zeros() as usize;
        Self {
            polynomial: Polynomial::new(evaluations),
            rounds,
        }
    }

    fn final_value(&self) -> Fr {
        self.polynomial.evals()[0]
    }
}

impl ProveRounds<Fr> for LinearSum {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.polynomial
                .bind_with_order(challenge, BindingOrder::HighToLow);
        }
        let half = self.polynomial.len() / 2;
        let at_zero: Fr = self.polynomial.evals()[..half].iter().copied().sum();
        let at_one: Fr = self.polynomial.evals()[half..].iter().copied().sum();
        if at_zero + at_one != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: at_zero + at_one,
            });
        }
        Ok(UnivariatePoly::from_evals(&[at_zero, at_one]))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.polynomial
            .bind_with_order(bind, BindingOrder::HighToLow);
        Ok(())
    }
}

#[test]
fn head_and_tail_aligned_members_share_a_stage() {
    let full_values: Vec<Fr> = (0..1 << 12)
        .map(|index| Fr::from_u64(index as u64 + 1))
        .collect();
    let head_values: Vec<Fr> = (0..1 << 10)
        .map(|index| Fr::from_u64(index as u64 * 3 + 2))
        .collect();
    let tail_values: Vec<Fr> = (0..1 << 10)
        .map(|index| Fr::from_u64(index as u64 * 5 + 4))
        .collect();
    let input_claims = [
        full_values.iter().copied().sum(),
        head_values.iter().copied().sum(),
        tail_values.iter().copied().sum(),
    ];
    let mut full = LinearSum::new(full_values.clone());
    let mut head = LinearSum::new(head_values.clone());
    let mut tail = LinearSum::new(tail_values.clone());
    let mut members = [
        StageMember {
            prover: &mut full,
            input_claim: input_claims[0],
            degree: 1,
            offset: 0,
        },
        StageMember {
            prover: &mut head,
            input_claim: input_claims[1],
            degree: 1,
            offset: 0,
        },
        StageMember {
            prover: &mut tail,
            input_claim: input_claims[2],
            degree: 1,
            offset: 2,
        },
    ];
    let mut prover_transcript = Blake3Transcript::<Fr>::new(b"scaled-stage-test");
    let (proof, proved) = prove_stage(&mut members, &mut prover_transcript).expect("prove stage");
    assert_eq!(
        proved.output_claims,
        vec![full.final_value(), head.final_value(), tail.final_value()]
    );

    let specs = [
        StageMemberSpec {
            rounds: 12,
            degree: 1,
            offset: 0,
        },
        StageMemberSpec {
            rounds: 10,
            degree: 1,
            offset: 0,
        },
        StageMemberSpec {
            rounds: 10,
            degree: 1,
            offset: 2,
        },
    ];
    let expected = [
        Polynomial::new(full_values),
        Polynomial::new(head_values),
        Polynomial::new(tail_values),
    ];
    let mut verifier_transcript = Blake3Transcript::<Fr>::new(b"scaled-stage-test");
    let verified = verify_stage_with(
        &proof,
        &specs,
        &input_claims,
        &mut verifier_transcript,
        |result| {
            expected
                .iter()
                .enumerate()
                .map(|(member, polynomial)| {
                    Ok(polynomial.evaluate(result.member_point(member, &specs)?))
                })
                .collect::<Result<Vec<_>, StreamError>>()
        },
    )
    .expect("verify stage");
    assert_eq!(verified.output_claims, proved.output_claims);
    assert_eq!(
        verified.member_point(1, &specs).expect("head point"),
        &verified.point[..10]
    );
    assert_eq!(
        verified.member_point(2, &specs).expect("tail point"),
        &verified.point[2..]
    );

    let mut tampered_transcript = Blake3Transcript::<Fr>::new(b"scaled-stage-test");
    assert!(verify_stage_with(
        &proof,
        &specs,
        &input_claims,
        &mut tampered_transcript,
        |result| {
            let mut outputs: Vec<Fr> = expected
                .iter()
                .enumerate()
                .map(|(member, polynomial)| {
                    polynomial.evaluate(result.member_point(member, &specs).expect("member point"))
                })
                .collect();
            outputs[1] += Fr::from_u64(1);
            Ok(outputs)
        },
    )
    .is_err());
}

#[test]
fn unimplemented_bound_values_reject_empty_output() {
    let member = LinearSum::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
    assert!(matches!(
        member.append_bound_values(&mut Vec::new()),
        Err(SumcheckError::MissingEvaluationSource {
            kind: "bound columns"
        })
    ));
}
