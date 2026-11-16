#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
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
use jolt_witness::{ProgramSource, TraceBackend};

use super::*;
use crate::optimized::booleanity::testing::{
    mixed_rows, with_booleanity_backend, with_trace_backend,
};
use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
use crate::optimized::streaming::DenseCap;
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

type CycleSlot = dyn PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>;

fn cycle_parity(log_t: usize, log_k_chunk: u8, kernel: &CycleSlot) {
    with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
        cycle_parity_on(backend, base_dimensions, kernel);
    });
}

/// The cycle-phase relation the parity cases prove over a backend, with its
/// inputs.
struct CycleCase {
    relation: BytecodeReadRafCycle<Fr>,
    challenges: BytecodeReadRafCyclePhaseCommittedChallenges<Fr>,
    claims: BytecodeReadRafInputClaims<Fr>,
    points: BytecodeReadRafInputClaims<Vec<Fr>>,
}

impl CycleCase {
    fn new(backend: &TraceBackend, base_dimensions: BooleanityDimensions) -> Self {
        let log_t = base_dimensions.log_t;
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
            committed_chunk_bits: base_dimensions.log_k_chunk,
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
        Self {
            relation,
            challenges: BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: Fr::from_u64(31),
            },
            claims: BytecodeReadRafInputClaims::<Fr>::default(),
            points: BytecodeReadRafInputClaims::<Vec<Fr>>::default(),
        }
    }

    fn inputs(&self) -> ProverInputs<'_, Fr, BytecodeReadRafCycle<Fr>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// The cap the streaming cycle kernel's dense tables, its RA factors, its
/// two coefficient tables and the fused increment, first fit together after
/// `depth` binds.
fn cap_at(relation: &BytecodeReadRafCycle<Fr>, depth: usize) -> DenseCap {
    let tables = relation.dimensions().num_committed_ra_polys() + 3;
    DenseCap {
        bytes: tables * size_of::<Fr>(),
        per_rows: 1 << depth,
    }
}

/// The streaming cycle kernel under [`cap_at`] this many binds.
struct StreamingFilledAt(usize);

impl PrepareKernel<Fr, BytecodeReadRafCycle<Fr>> for StreamingFilledAt {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, BytecodeReadRafCycle<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = BytecodeReadRafCycle<Fr>>>, KernelError<Fr>>
    {
        let cap = cap_at(inputs.relation, self.0);
        Ok(Box::new(CycleKernel::streaming(witness, &inputs, cap)?))
    }
}

fn cycle_parity_on(
    backend: &TraceBackend,
    base_dimensions: BooleanityDimensions,
    kernel: &CycleSlot,
) {
    let case = CycleCase::new(backend, base_dimensions);
    let mut session = ProofSession::default();
    let mut reference = ReferenceBackend
        .prepare(&mut session, backend, case.inputs())
        .unwrap();
    let mut optimized = kernel
        .prepare(&mut session, backend, case.inputs())
        .unwrap();

    let claim = probe_input_claim(reference.as_mut());
    run_lockstep(
        reference.as_mut(),
        optimized.as_mut(),
        claim,
        &synthetic_point(base_dimensions.log_t, 211),
    );
    assert_eq!(
        reference.output_claims(&case.claims).unwrap(),
        optimized.output_claims(&case.claims).unwrap()
    );
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
    cycle_parity(2, 4, &OptimizedBytecodeReadRafCycle);
}

#[test]
fn bytecode_cycle_with_padding_matches_reference_through_materialization() {
    for chunk in [4, 8] {
        cycle_parity(6, chunk, &OptimizedBytecodeReadRafCycle);
    }
}

#[test]
fn bytecode_cycle_matches_reference_k256() {
    cycle_parity(3, 8, &OptimizedBytecodeReadRafCycle);
}

/// 4 of 4 cycles leave no padding suffix and 4 of 64 leave one; 1,500 of
/// 2,048 and 6,000 of 8,192 leave one that starts inside a pair row of the
/// later streamed rounds, and 2,048 of 2,048 fill their domain; an empty
/// trace is all suffix, and the kernel still visits cycle 0's pair row, for
/// its entry term. The kernel fills its dense tables at the bind the
/// production cap selects, the last on a shorter domain, and under caps that
/// select the fourth bind, one between and the last two, which leave two
/// entries and one per table. Every streamed round gathers the fused
/// increment from the resident rows.
#[test]
fn streaming_cycle_matches_reference_over_resident_rows() {
    for (log_t, chunk) in [(2, 4), (6, 4), (6, 8)] {
        cycle_parity(log_t, chunk, &StreamingBytecodeReadRafCycle);
        for depth in 4..=log_t {
            cycle_parity(log_t, chunk, &StreamingFilledAt(depth));
        }
    }
    for (log_t, rows) in [
        (11, mixed_rows(1500)),
        (11, mixed_rows(2048)),
        (13, mixed_rows(6000)),
        (2, Vec::new()),
    ] {
        with_trace_backend(log_t, 4, rows, |backend, base_dimensions| {
            cycle_parity_on(backend, base_dimensions, &StreamingBytecodeReadRafCycle);
            if log_t == 11 {
                for depth in [4, 7, log_t - 1, log_t] {
                    cycle_parity_on(backend, base_dimensions, &StreamingFilledAt(depth));
                }
            }
        });
    }
}

/// A one-cycle domain has no round: the streaming cycle kernel holds its RA
/// factor dense from the start, takes the fused increment of its cycle, a
/// load that changes `rd`, as the rows' suffix value, and opens the reference
/// kernel's claims.
#[test]
fn streaming_cycle_serves_a_one_cycle_domain() {
    with_trace_backend(0, 4, mixed_rows(1), |backend, base_dimensions| {
        let case = CycleCase::new(backend, base_dimensions);
        let mut session = ProofSession::default();
        let slots: [&CycleSlot; 2] = [&ReferenceBackend, &StreamingBytecodeReadRafCycle];
        let [reference, streaming] = slots.map(|slot| {
            slot.prepare(&mut session, backend, case.inputs())
                .unwrap()
                .output_claims(&case.claims)
                .unwrap()
        });
        assert_eq!(reference, streaming);
    });
}

/// The streaming cycle kernel holds one coefficient in each coefficient
/// table, cycle 0's, and no table by cycle until the first bind at which its
/// RA factors, its two coefficient tables and the fused increment fit its
/// cap together: a cap one byte short of them after six binds fills at the
/// seventh. Over a padded trace that bind fills the tables for the bound
/// indices that hold a cycle before the padding.
#[test]
fn streaming_cycle_holds_no_table_until_its_tables_fit_the_cap() {
    with_trace_backend(11, 4, mixed_rows(1500), |backend, base_dimensions| {
        let case = CycleCase::new(backend, base_dimensions);
        let fits = cap_at(&case.relation, 6);
        let short = DenseCap {
            bytes: fits.bytes - 1,
            ..fits
        };
        for (cap, depth) in [(fits, 6), (short, 7)] {
            let mut kernel = CycleKernel::streaming(backend, &case.inputs(), cap).unwrap();
            for round in 0..depth {
                assert!(kernel.rows.lazy().is_none());
                assert_eq!(kernel.combined.prefix_len(), 1);
                assert_eq!(kernel.fused_combined.prefix_len(), 1);
                kernel.bind(Fr::from_u64(31 + round as u64)).unwrap();
            }
            let filled = 1500usize.div_ceil(1 << depth);
            let rows = kernel.rows.lazy().unwrap();
            assert_eq!(rows.ra.prefix_len(), filled);
            assert!(
                matches!(&rows.fused_inc, LazyFusedInc::Prefix { values, .. } if values.len() == filled)
            );
            assert_eq!(kernel.combined.prefix_len(), filled);
            assert_eq!(kernel.fused_combined.prefix_len(), filled);
        }
    });
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
