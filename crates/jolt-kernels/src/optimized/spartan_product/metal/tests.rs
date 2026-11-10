#![expect(clippy::unwrap_used, reason = "independent reference oracle")]
use super::super::tests::uniskip_input_claims;
use super::*;
use crate::optimized::registers_read_write::test_support::{
    assert_kernel_parity_with_session, structured_fixture,
};
use crate::reference::spartan_product::{ReferenceProductRemainder, SpartanProductKernel};
use crate::ReferenceBackend;
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
    InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputChallenges, InstructionInputInputClaims,
};
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial, TraceDimensions};
use jolt_field::One;
use jolt_poly::Polynomial;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage2::product_remainder::product_remainder_input_values_from_uniskip_output;
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::{collect_bundles, JoltWitnessOracle};

#[test]
fn resident_operands_share_reference_claims_and_input_handoff() {
    let log_t = 17;
    let slot = MetalProductUniskip::new().unwrap();
    let icr_slot = slot.claim_reduction();
    let input_slot = slot.input().unwrap();
    let mut tau: Vec<_> = (0..log_t)
        .map(|i| AkitaField::from_u64(31 + 7 * i as u64))
        .collect();
    tau[log_t - 1] = AkitaField::zero();
    tau[log_t - 2] = AkitaField::one();
    let challenges: Vec<_> = (0..log_t)
        .map(|i| AkitaField::from_u64(if i < 2 { i as u64 } else { 73 + 11 * i as u64 }))
        .collect();
    structured_fixture((1 << 16) + 1).with_plane(log_t, |witness| {
        let rows: Vec<SpartanProductRow> = collect_bundles(witness, 1 << log_t).unwrap();
        let uniskip_inputs = uniskip_input_claims::<AkitaField>(
            &rows,
            #[cfg(feature = "field-inline")]
            &[],
            &tau,
        );
        let mut reference_session = ProofSession::default();
        reference_session
            .park(SpartanProductKernel::<AkitaField>::prepare(log_t, &tau, witness).unwrap());
        let mut session = ProofSession::default();
        slot.prepare(&mut session, log_t, &tau, witness).unwrap();
        let weak_stage2 = Arc::downgrade(&session.state::<ForClaimReduction>().unwrap().0);
        let weak_core = Arc::downgrade(&session.state::<ForInput>().unwrap().0);
        let r0 = AkitaField::from_u64(101);
        let tau_high = AkitaField::from_u64(59);
        let expected = ReferenceBackend
            .first_round_poly(&mut reference_session, &[tau_high], &uniskip_inputs)
            .unwrap();
        let actual = slot
            .first_round_poly(&mut session, &[tau_high], &uniskip_inputs)
            .unwrap();
        assert_eq!(actual, expected);
        let mut previous = expected.evaluate(r0);
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            r0,
            tau_high,
            tau.clone(),
        );
        let claims = product_remainder_input_values_from_uniskip_output(previous);
        let points = SumcheckInputPoints::<AkitaField, ProductRemainder<AkitaField>>::default();
        let no_challenges = NoChallenges::<AkitaField>::default();
        let inputs = ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &no_challenges,
        };
        let mut reference = ReferenceProductRemainder
            .prepare(
                &mut reference_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();
        let mut metal = ProductRemainderKernel::prepare(session.take().unwrap(), &inputs).unwrap();
        assert!(matches!(metal.columns, ProductColumns::Device(_)));
        for (round, &challenge) in challenges.iter().enumerate() {
            let bind = (round > 0).then(|| challenges[round - 1]);
            let expected = reference.prove_round(bind, round, previous).unwrap();
            assert_eq!(
                metal.prove_round(bind, round, previous).unwrap(),
                expected,
                "Product round {round}"
            );
            previous = expected.evaluate(challenge);
        }
        reference.finish_rounds(challenges[log_t - 1]).unwrap();
        metal.finish_rounds(challenges[log_t - 1]).unwrap();
        assert!(matches!(metal.columns, ProductColumns::Cpu { .. }));
        let output = metal.output_claims(&claims).unwrap();
        assert_eq!(output, reference.output_claims(&claims).unwrap());
        #[cfg(feature = "field-inline")]
        let output = output.base;
        let bound_point: Vec<_> = challenges.iter().rev().copied().collect();
        let shared = ResidentProduct {
            gpu: Arc::clone(&slot.0),
            source: weak_stage2.upgrade().unwrap(),
        };
        assert!(
            shared.openings(&tau).is_err(),
            "claim cache must reject a different point"
        );
        assert!(shared.openings(&bound_point).is_ok());
        drop(shared);
        let output_points = relation
            .derive_opening_points(&challenges, &points)
            .unwrap();
        metal
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .unwrap();
        drop(metal);
        drop(reference);

        let evaluate = |polynomial| {
            let values = JoltWitnessOracle::<AkitaField>::oracle_table(
                witness,
                JoltPolynomialId::Virtual(polynomial),
            )
            .unwrap();
            Polynomial::new(values).evaluate(&tau)
        };
        let claims = InstructionClaimReductionInputClaims {
            lookup_output: evaluate(JoltVirtualPolynomial::LookupOutput),
            left_lookup_operand: evaluate(JoltVirtualPolynomial::LeftLookupOperand),
            right_lookup_operand: evaluate(JoltVirtualPolynomial::RightLookupOperand),
            left_instruction_input: evaluate(JoltVirtualPolynomial::LeftInstructionInput),
            right_instruction_input: evaluate(JoltVirtualPolynomial::RightInstructionInput),
        };
        let points = InstructionClaimReductionInputClaims {
            lookup_output: tau.clone(),
            left_lookup_operand: tau.clone(),
            right_lookup_operand: tau.clone(),
            left_instruction_input: tau.clone(),
            right_instruction_input: tau.clone(),
        };
        let relation = InstructionClaimReduction::new(TraceDimensions::new(log_t), tau.clone());
        let gamma = InstructionClaimReductionChallenges {
            gamma: AkitaField::from_u64(193),
        };
        assert_kernel_parity_with_session(
            &mut session,
            &icr_slot,
            witness,
            &relation,
            &claims,
            &points,
            &gamma,
            relation.input_claim(&claims, &gamma).unwrap(),
            &challenges,
        );
        assert!(weak_stage2.upgrade().is_none());
        assert!(weak_core.upgrade().is_some());
        let relation = InstructionInput::new(TraceDimensions::new(log_t), bound_point.clone());
        let claims = InstructionInputInputClaims {
            left_instruction_input: output.left_instruction_input,
            right_instruction_input: output.right_instruction_input,
        };
        let points = InstructionInputInputClaims {
            left_instruction_input: bound_point.clone(),
            right_instruction_input: bound_point,
        };
        let gamma = InstructionInputChallenges {
            gamma: AkitaField::from_u64(223),
        };
        assert_kernel_parity_with_session(
            &mut session,
            &input_slot,
            witness,
            &relation,
            &claims,
            &points,
            &gamma,
            relation.input_claim(&claims, &gamma).unwrap(),
            &challenges,
        );
        assert!(session.state::<ForInput>().is_none());
        assert!(weak_core.upgrade().is_none());
    });
}
