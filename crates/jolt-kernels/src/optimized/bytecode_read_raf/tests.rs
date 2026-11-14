#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::{Fr, Ring};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeFold;
use jolt_verifier::stages::relations::SumcheckInputPoints;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6a::field_inline::FieldInlineBytecodeReadRafGeometry;
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
    BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
};
use jolt_witness::testing::with_sample_backend;
use jolt_witness::{JoltWitnessOracle, ProgramSource, TraceBackend};

use super::super::instruction_read_raf::{SharedInstructionRows, SharedInstructionRowsWeak};
use super::*;
#[cfg(feature = "field-inline")]
use crate::optimized::field_registers_testing::structured_field_register_fixture;
use crate::optimized::parity::{
    probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
};
use crate::ReferenceBackend;

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn run_pair(committed_program: bool) {
    with_sample_backend(|backend| run_pair_on(backend, committed_program));
}

/// Parity with field-inline enabled over a program containing field instructions:
/// the field-inline legs of both phases carry non-zero terms, so a drift between
/// the reference and optimized field-inline folds surfaces here rather than only at
/// the e2e.
#[cfg(feature = "field-inline")]
fn run_pair_with_field_inline_program(f: impl FnOnce(&TraceBackend)) {
    structured_field_register_fixture(12).with_plane(4, |backend| {
        f(backend);
    });
}

fn run_pair_on(backend: &TraceBackend, committed_program: bool) {
    {
        let log_t = JoltWitnessOracle::<Fr>::shape(
            backend,
            JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
        )
        .unwrap()
        .rows()
        .ilog2() as usize;
        let (bytecode_d, chunk_bits) =
            probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
        assert!(bytecode_d > 0, "fixture serves no BytecodeRa polynomials");
        let program = backend.program_preprocessing();
        let bytecode_len = program.bytecode.bytecode.len();
        assert!(bytecode_len.is_power_of_two());
        let log_k = bytecode_len.ilog2() as usize;
        let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, bytecode_d);
        let entry_bytecode_index = 1.min(bytecode_len - 1);

        let stage_cycle_points: [Vec<Fr>; 5] =
            std::array::from_fn(|s| synthetic_point(log_t, 11 + s as u64));
        let stage_points = BytecodeStagePoints {
            stage_cycle_points: stage_cycle_points.clone(),
            register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 31),
            register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 37),
            fused_inc_cycle_points: Vec::new(),
        };
        #[cfg(feature = "field-inline")]
        let field_read_write_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 41);
        #[cfg(feature = "field-inline")]
        let field_val_evaluation_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 43);

        let address_relation = BytecodeReadRafAddressPhase::new(
            dimensions,
            committed_program,
            stage_points.clone(),
            entry_bytecode_index,
        );
        #[cfg(feature = "field-inline")]
        let address_relation =
            address_relation.with_field_inline_geometry(FieldInlineBytecodeReadRafGeometry {
                read_write_point: field_read_write_point.clone(),
                val_evaluation_point: field_val_evaluation_point.clone(),
            });
        let address_challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma: fr(3),
            stage1_gamma: fr(5),
            stage2_gamma: fr(7),
            stage3_gamma: fr(11),
            stage4_gamma: fr(13),
            stage5_gamma: fr(17),
        };
        let address_claims = SumcheckInputClaims::<Fr, BytecodeReadRafAddressPhase<Fr>>::default();
        let address_input_points =
            SumcheckInputPoints::<Fr, BytecodeReadRafAddressPhase<Fr>>::default();

        let mut session = ProofSession::default();
        let mut reference =
            <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>>>::prepare(
                &ReferenceBackend,
                &mut session,
                backend,
                ProverInputs {
                    relation: &address_relation,
                    claims: &address_claims,
                    points: &address_input_points,
                    challenges: &address_challenges,
                },
            )
            .unwrap();
        let mut optimized = OptimizedBytecodeReadRafAddress
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &address_relation,
                    claims: &address_claims,
                    points: &address_input_points,
                    challenges: &address_challenges,
                },
            )
            .unwrap();
        assert!(
            session.state::<SharedInstructionRows>().is_some()
                || session.state::<SharedInstructionRowsWeak>().is_some(),
            "the address phase must park the shared cycle rows"
        );

        let claim = probe_input_claim(reference.as_mut());
        assert!(
            claim != Fr::from_u64(0),
            "bytecode parity fixture must exercise a nonzero relation"
        );
        let address_sumcheck_challenges = synthetic_point(log_k, 101);
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &address_sumcheck_challenges,
        );
        let reference_outputs = reference.output_claims(&address_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&address_claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);
        assert_eq!(
            reference_outputs.val_stages.len(),
            if committed_program { 5 } else { 0 }
        );

        // ---- Stage 6b: cycle phase, from the same session (the parked
        // cycle rows are reused, mirroring the production 6a→6b flow) and
        // the production r_address wiring (the reversed 6a point).
        let r_address: Vec<Fr> = address_sumcheck_challenges.iter().rev().copied().collect();
        let stage_gammas = address_challenges.stage_gamma_powers();
        let cycle_relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
            dimensions,
            r_address,
            stage_cycle_points: stage_cycle_points.clone(),
            entry_bytecode_index,
            committed_chunk_bits: chunk_bits,
            table_fold: Some(BytecodeReadRafTableFoldInputs {
                bytecode: &program.bytecode.bytecode,
                register_read_write_point: &stage_points.register_read_write_point
                    [..REGISTER_ADDRESS_BITS],
                register_val_evaluation_point: &stage_points.register_val_evaluation_point
                    [..REGISTER_ADDRESS_BITS],
                stage_gammas: std::array::from_fn(|s| stage_gammas[s].as_slice()),
            }),
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineBytecodeFold {
                read_write_address: field_read_write_point[..FIELD_REGISTERS_LOG_K].to_vec(),
                read_write_cycle: field_read_write_point[FIELD_REGISTERS_LOG_K..].to_vec(),
                val_evaluation_address: field_val_evaluation_point[..FIELD_REGISTERS_LOG_K]
                    .to_vec(),
                val_evaluation_cycle: field_val_evaluation_point[FIELD_REGISTERS_LOG_K..].to_vec(),
                gammas:
                    jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                        &address_challenges,
                    ),
            },
        })
        .unwrap();
        let cycle_challenges = BytecodeReadRafCyclePhaseCommittedChallenges { gamma: fr(19) };
        let cycle_claims = BytecodeReadRafInputClaims::<Fr>::default();
        let cycle_input_points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();

        let mut reference =
            <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>>::prepare(
                &ReferenceBackend,
                &mut session,
                backend,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_input_points,
                    challenges: &cycle_challenges,
                },
            )
            .unwrap();
        let mut optimized = OptimizedBytecodeReadRafCycle
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_input_points,
                    challenges: &cycle_challenges,
                },
            )
            .unwrap();

        let claim = probe_input_claim(reference.as_mut());
        assert!(
            claim != Fr::from_u64(0),
            "bytecode parity fixture must exercise a nonzero relation"
        );
        let cycle_sumcheck_challenges = synthetic_point(log_t, 211);
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &cycle_sumcheck_challenges,
        );
        assert_eq!(
            reference.output_claims(&cycle_claims).unwrap(),
            optimized.output_claims(&cycle_claims).unwrap()
        );
    }
}

#[test]
fn bytecode_phases_match_reference_through_a_shared_session() {
    run_pair(false);
}

#[test]
fn bytecode_phases_match_reference_in_committed_program_mode() {
    run_pair(true);
}

#[cfg(feature = "field-inline")]
#[test]
fn bytecode_phases_match_reference_with_field_inline_instructions() {
    run_pair_with_field_inline_program(|backend| run_pair_on(backend, false));
}
