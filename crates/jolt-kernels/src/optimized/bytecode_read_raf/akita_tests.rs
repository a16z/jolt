#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
use jolt_claims::protocols::jolt::geometry::bytecode::{
    BytecodeReadRafDimensions, LATTICE_FUSED_INC_STAGES,
};
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
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
use jolt_witness::ProgramSource;

use super::*;
use crate::optimized::booleanity::testing::with_booleanity_backend;
use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
use crate::ReferenceBackend;

type AddressSlot = dyn PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>>;

fn address_parity(log_t: usize, log_k_chunk: u8, committed_program: bool, kernel: &AddressSlot) {
    with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
        let bytecode_len = backend.program_preprocessing().bytecode.bytecode.len();
        let dimensions = BytecodeReadRafDimensions::new(
            log_t,
            bytecode_len.ilog2() as usize,
            base_dimensions.layout.bytecode(),
        );
        address_parity_on(backend, dimensions, committed_program, kernel);
    });
}

fn address_parity_on(
    backend: &dyn JoltWitnessPlane<Fr>,
    dimensions: BytecodeReadRafDimensions,
    committed_program: bool,
    kernel: &AddressSlot,
) {
    let log_t = dimensions.log_t();
    let relation = BytecodeReadRafAddressPhase::new(
        dimensions,
        committed_program,
        BytecodeStagePoints {
            stage_cycle_points: std::array::from_fn(|stage| {
                synthetic_point(log_t, 11 + stage as u64)
            }),
            register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 23),
            register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 29),
            fused_inc_cycle_points: (0..LATTICE_FUSED_INC_STAGES)
                .map(|stage| synthetic_point(log_t, 41 + stage as u64))
                .collect(),
        },
        0,
    );
    // The relation requires field-inline geometry even for inactive fixtures.
    #[cfg(feature = "field-inline")]
    let relation = relation.with_field_inline_geometry(FieldInlineBytecodeReadRafGeometry {
        read_write_point: synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 61),
        val_evaluation_point: synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 67),
    });
    let challenges = BytecodeReadRafAddressPhaseChallenges {
        gamma: Fr::from_u64(3),
        stage1_gamma: Fr::from_u64(5),
        stage2_gamma: Fr::from_u64(7),
        stage3_gamma: Fr::from_u64(11),
        stage4_gamma: Fr::from_u64(13),
        stage5_gamma: Fr::from_u64(17),
    };
    let claims = SumcheckInputClaims::<Fr, BytecodeReadRafAddressPhase<Fr>>::default();
    let input_points = SumcheckInputPoints::<Fr, BytecodeReadRafAddressPhase<Fr>>::default();

    let mut session = ProofSession::default();
    let mut reference = ReferenceBackend
        .prepare(
            &mut session,
            backend,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &input_points,
                challenges: &challenges,
            },
        )
        .unwrap();
    let mut optimized = kernel
        .prepare(
            &mut session,
            backend,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &input_points,
                challenges: &challenges,
            },
        )
        .unwrap();

    let claim = probe_input_claim(reference.as_mut());
    assert!(
        claim != Fr::from_u64(0),
        "bytecode parity fixture must exercise a nonzero relation"
    );
    run_lockstep(
        reference.as_mut(),
        optimized.as_mut(),
        claim,
        &synthetic_point(dimensions.log_k(), 101),
    );
    let reference = reference.output_claims(&claims).unwrap();
    let optimized = optimized.output_claims(&claims).unwrap();
    assert_eq!(reference, optimized);
    assert_eq!(
        optimized.val_stages.len(),
        if committed_program {
            NUM_BYTECODE_VAL_STAGES
        } else {
            0
        }
    );
}

fn cycle_parity(log_t: usize, log_k_chunk: u8) {
    with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
        let program = backend.program_preprocessing();
        let bytecode_len = program.bytecode.bytecode.len();
        let dimensions = BytecodeReadRafDimensions::new(
            log_t,
            bytecode_len.ilog2() as usize,
            base_dimensions.layout.bytecode(),
        );
        let stage_cycle_points =
            std::array::from_fn(|stage| synthetic_point(log_t, 11 + stage as u64));
        let address_challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma: Fr::from_u64(3),
            stage1_gamma: Fr::from_u64(5),
            stage2_gamma: Fr::from_u64(7),
            stage3_gamma: Fr::from_u64(11),
            stage4_gamma: Fr::from_u64(13),
            stage5_gamma: Fr::from_u64(17),
        };
        let stage_gammas = address_challenges.stage_gamma_powers();
        #[cfg(feature = "field-inline")]
        let field_read_write_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 41);
        #[cfg(feature = "field-inline")]
        let field_val_evaluation_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 43);
        let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
            dimensions,
            r_address: synthetic_point(dimensions.log_k(), 19),
            stage_cycle_points,
            entry_bytecode_index: 0,
            committed_chunk_bits: usize::from(log_k_chunk),
            table_fold: Some(BytecodeReadRafTableFoldInputs {
                bytecode: &program.bytecode.bytecode,
                register_read_write_point: &synthetic_point(REGISTER_ADDRESS_BITS, 23),
                register_val_evaluation_point: &synthetic_point(REGISTER_ADDRESS_BITS, 29),
                stage_gammas: std::array::from_fn(|stage| stage_gammas[stage].as_slice()),
            }),
            // All-inactive and well-formed, mirroring `run_pair`: the composed
            // reference cycle kernel folds the field-inline rows (all zero) at
            // these points, so parity with the optimized kernel holds.
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
        let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: Fr::from_u64(31),
        };
        let claims = BytecodeReadRafInputClaims::<Fr>::default();
        let input_points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();

        let mut session = ProofSession::default();
        let mut reference = ReferenceBackend
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &input_points,
                    challenges: &challenges,
                },
            )
            .unwrap();
        let mut optimized = OptimizedBytecodeReadRafCycle
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &input_points,
                    challenges: &challenges,
                },
            )
            .unwrap();

        let claim = probe_input_claim(reference.as_mut());
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &synthetic_point(log_t, 211),
        );
        assert_eq!(
            reference.output_claims(&claims).unwrap(),
            optimized.output_claims(&claims).unwrap()
        );
    });
}

#[cfg(feature = "field-inline")]
fn address_parity_with_active_field_inline(kernel: &AddressSlot) {
    use crate::optimized::field_registers_testing::structured_field_register_fixture;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_witness::JoltWitnessOracle;

    structured_field_register_fixture(24).with_plane(5, |backend| {
        let bytecode_len = backend.program_preprocessing().bytecode.bytecode.len();
        let bytecode_chunks = (0..64)
            .take_while(|index| {
                JoltWitnessOracle::<Fr>::shape(
                    backend,
                    JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(*index)),
                )
                .is_ok()
            })
            .count();
        let dimensions =
            BytecodeReadRafDimensions::new(5, bytecode_len.ilog2() as usize, bytecode_chunks);
        address_parity_on(backend, dimensions, true, kernel);
    });
}

#[cfg(feature = "field-inline")]
#[test]
fn bytecode_address_matches_reference_with_active_field_inline() {
    address_parity_with_active_field_inline(&OptimizedBytecodeReadRafAddress);
}

#[cfg(feature = "field-inline")]
#[test]
fn streaming_address_matches_reference_with_active_field_inline() {
    address_parity_with_active_field_inline(&StreamingBytecodeReadRafAddress);
}

#[test]
fn bytecode_cycle_matches_reference_k16() {
    cycle_parity(2, 4);
}

#[test]
fn bytecode_cycle_with_padding_matches_reference_through_materialization() {
    for chunk in [4, 8] {
        cycle_parity(6, chunk);
    }
}

#[test]
fn bytecode_cycle_matches_reference_k256() {
    cycle_parity(3, 8);
}

#[test]
fn bytecode_address_matches_reference_k16() {
    address_parity(2, 4, false, &OptimizedBytecodeReadRafAddress);
}

#[test]
fn bytecode_address_matches_reference_k256_committed() {
    address_parity(3, 8, true, &OptimizedBytecodeReadRafAddress);
}

/// 4 of 64 cycles leave a padding suffix after a walk that ends inside a
/// block; 4 of 4 leave none.
#[test]
fn streaming_address_matches_reference_over_resident_rows() {
    for (log_t, committed_program) in [(6, true), (2, false)] {
        address_parity(
            log_t,
            4,
            committed_program,
            &StreamingBytecodeReadRafAddress,
        );
    }
}
