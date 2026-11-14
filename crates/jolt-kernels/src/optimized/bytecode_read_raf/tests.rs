#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_dory::DoryScheme;
use jolt_field::{Fr, Ring};
use jolt_riscv::JoltTraceRow;
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
use jolt_witness::{JoltVmWitnessInputs, JoltWitnessOracle, ProgramSource, TraceBackend};

use super::super::instruction_read_raf::{SharedInstructionRows, SharedInstructionRowsWeak};
use super::*;
use crate::optimized::booleanity::testing::{alu_row, load_row, mixed_rows, with_trace_backend};
#[cfg(feature = "field-inline")]
use crate::optimized::field_registers_testing::structured_field_register_fixture;
use crate::optimized::parity::{
    probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
};
use crate::optimized::registers_read_write::test_support::structured_fixture;
use crate::optimized::streaming::StreamingFamilies;
use crate::{JoltBackend, ReferenceBackend};

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn run_pair(committed_program: bool) {
    with_sample_backend(|backend| run_pair_on(backend, committed_program, false));
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

/// The address-phase relation `run_pair_on` proves over a backend, with its
/// inputs.
struct AddressCase {
    dimensions: BytecodeReadRafDimensions,
    chunk_bits: usize,
    entry_bytecode_index: usize,
    stage_points: BytecodeStagePoints<Fr>,
    #[cfg(feature = "field-inline")]
    field_read_write_point: Vec<Fr>,
    #[cfg(feature = "field-inline")]
    field_val_evaluation_point: Vec<Fr>,
    relation: BytecodeReadRafAddressPhase<Fr>,
    challenges: BytecodeReadRafAddressPhaseChallenges<Fr>,
    claims: SumcheckInputClaims<Fr, BytecodeReadRafAddressPhase<Fr>>,
    points: SumcheckInputPoints<Fr, BytecodeReadRafAddressPhase<Fr>>,
}

impl AddressCase {
    fn new(backend: &TraceBackend, committed_program: bool) -> Self {
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

        let stage_points = BytecodeStagePoints {
            stage_cycle_points: std::array::from_fn(|s| synthetic_point(log_t, 11 + s as u64)),
            register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 31),
            register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 37),
            fused_inc_cycle_points: Vec::new(),
        };
        #[cfg(feature = "field-inline")]
        let field_read_write_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 41);
        #[cfg(feature = "field-inline")]
        let field_val_evaluation_point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, 43);

        let relation = BytecodeReadRafAddressPhase::new(
            dimensions,
            committed_program,
            stage_points.clone(),
            entry_bytecode_index,
        );
        #[cfg(feature = "field-inline")]
        let relation = relation.with_field_inline_geometry(FieldInlineBytecodeReadRafGeometry {
            read_write_point: field_read_write_point.clone(),
            val_evaluation_point: field_val_evaluation_point.clone(),
        });
        Self {
            dimensions,
            chunk_bits,
            entry_bytecode_index,
            stage_points,
            #[cfg(feature = "field-inline")]
            field_read_write_point,
            #[cfg(feature = "field-inline")]
            field_val_evaluation_point,
            relation,
            challenges: BytecodeReadRafAddressPhaseChallenges {
                gamma: fr(3),
                stage1_gamma: fr(5),
                stage2_gamma: fr(7),
                stage3_gamma: fr(11),
                stage4_gamma: fr(13),
                stage5_gamma: fr(17),
            },
            claims: SumcheckInputClaims::<Fr, BytecodeReadRafAddressPhase<Fr>>::default(),
            points: SumcheckInputPoints::<Fr, BytecodeReadRafAddressPhase<Fr>>::default(),
        }
    }

    fn inputs(&self) -> ProverInputs<'_, Fr, BytecodeReadRafAddressPhase<Fr>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// The cycle-phase relation that follows an [`AddressCase`] bound to
/// `r_address`, with its inputs.
struct CycleCase {
    relation: BytecodeReadRafCycle<Fr>,
    challenges: BytecodeReadRafCyclePhaseCommittedChallenges<Fr>,
    claims: BytecodeReadRafInputClaims<Fr>,
    points: BytecodeReadRafInputClaims<Vec<Fr>>,
}

impl CycleCase {
    fn new(backend: &TraceBackend, address: &AddressCase, r_address: Vec<Fr>) -> Self {
        let stage_gammas = address.challenges.stage_gamma_powers();
        let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
            dimensions: address.dimensions,
            r_address,
            stage_cycle_points: address.stage_points.stage_cycle_points.clone(),
            entry_bytecode_index: address.entry_bytecode_index,
            committed_chunk_bits: address.chunk_bits,
            table_fold: Some(BytecodeReadRafTableFoldInputs {
                bytecode: &backend.program_preprocessing().bytecode.bytecode,
                register_read_write_point: &address.stage_points.register_read_write_point
                    [..REGISTER_ADDRESS_BITS],
                register_val_evaluation_point: &address.stage_points.register_val_evaluation_point
                    [..REGISTER_ADDRESS_BITS],
                stage_gammas: std::array::from_fn(|s| stage_gammas[s].as_slice()),
            }),
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineBytecodeFold {
                read_write_address: address.field_read_write_point[..FIELD_REGISTERS_LOG_K]
                    .to_vec(),
                read_write_cycle: address.field_read_write_point[FIELD_REGISTERS_LOG_K..].to_vec(),
                val_evaluation_address: address.field_val_evaluation_point[..FIELD_REGISTERS_LOG_K]
                    .to_vec(),
                val_evaluation_cycle: address.field_val_evaluation_point[FIELD_REGISTERS_LOG_K..]
                    .to_vec(),
                gammas:
                    jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                        &address.challenges,
                    ),
            },
        })
        .unwrap();
        Self {
            relation,
            challenges: BytecodeReadRafCyclePhaseCommittedChallenges { gamma: fr(19) },
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

fn session_holds_rows(session: &ProofSession) -> bool {
    session.state::<SharedInstructionRows>().is_some()
        || session.state::<SharedInstructionRowsWeak>().is_some()
}

/// `streaming` runs both phases on the streaming kernels.
fn run_pair_on(backend: &TraceBackend, committed_program: bool, streaming: bool) {
    {
        let case = AddressCase::new(backend, committed_program);
        let mut session = ProofSession::default();
        let mut reference = <ReferenceBackend as PrepareKernel<
            Fr,
            BytecodeReadRafAddressPhase<Fr>,
        >>::prepare(
            &ReferenceBackend, &mut session, backend, case.inputs()
        )
        .unwrap();
        let address: &dyn PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>> = if streaming {
            &StreamingBytecodeReadRafAddress
        } else {
            &OptimizedBytecodeReadRafAddress
        };
        let mut optimized = address
            .prepare(&mut session, backend, case.inputs())
            .unwrap();
        assert_eq!(
            session_holds_rows(&session),
            !streaming,
            "the default address phase parks the shared cycle rows; the streaming one collects none"
        );

        let claim = probe_input_claim(reference.as_mut());
        assert!(
            claim != Fr::from_u64(0),
            "bytecode parity fixture must exercise a nonzero relation"
        );
        let address_sumcheck_challenges = synthetic_point(case.dimensions.log_k(), 101);
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &address_sumcheck_challenges,
        );
        let reference_outputs = reference.output_claims(&case.claims).unwrap();
        let optimized_outputs = optimized.output_claims(&case.claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);
        assert_eq!(
            reference_outputs.val_stages.len(),
            if committed_program { 5 } else { 0 }
        );

        // ---- Stage 6b: cycle phase, from the same session (the parked
        // cycle rows are reused, mirroring the production 6a→6b flow) and
        // the production r_address wiring (the reversed 6a point).
        let r_address: Vec<Fr> = address_sumcheck_challenges.iter().rev().copied().collect();
        let cycle_case = CycleCase::new(backend, &case, r_address);
        let mut reference =
            <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>>::prepare(
                &ReferenceBackend,
                &mut session,
                backend,
                cycle_case.inputs(),
            )
            .unwrap();
        let cycle: &dyn PrepareKernel<Fr, BytecodeReadRafCycle<Fr>> = if streaming {
            &StreamingBytecodeReadRafCycle
        } else {
            &OptimizedBytecodeReadRafCycle
        };
        let mut optimized = cycle
            .prepare(&mut session, backend, cycle_case.inputs())
            .unwrap();
        assert_eq!(
            session_holds_rows(&session),
            !streaming,
            "the streaming cycle phase collects no cycle rows"
        );

        let claim = probe_input_claim(reference.as_mut());
        assert!(
            claim != Fr::from_u64(0),
            "bytecode parity fixture must exercise a nonzero relation"
        );
        let cycle_sumcheck_challenges = synthetic_point(case.dimensions.log_t(), 211);
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &cycle_sumcheck_challenges,
        );
        assert_eq!(
            reference.output_claims(&cycle_case.claims).unwrap(),
            optimized.output_claims(&cycle_case.claims).unwrap()
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
    run_pair_with_field_inline_program(|backend| run_pair_on(backend, false, false));
}

/// The sample trace, 363 of 512 cycles, 1,500 of 2,048 and 6,000 of 8,192
/// leave a padding suffix, which the address walk meets inside a block; 16
/// of 16 and 2,048 of 2,048 leave none; an empty trace is all suffix, and
/// the cycle phase still holds cycle 0, for its entry term. The cycle
/// phase's RA factors turn dense at the first bind on the sample trace, on
/// 16 cycles and on the empty trace, at the third on 512 cycles and at the
/// fourth on the longer traces: without doubling their branch tables for
/// that bind up to 2,048 cycles, and after doubling them on 8,192, as on
/// every long trace.
#[test]
fn streaming_phases_match_reference_over_resident_rows() {
    with_sample_backend(|backend| run_pair_on(backend, true, true));
    for (cycles, log_t) in [(363, 9), (16, 4)] {
        structured_fixture(cycles).with_plane(log_t, |backend| run_pair_on(backend, false, true));
    }
    for (log_t, rows) in [
        (11, mixed_rows(1500)),
        (11, mixed_rows(2048)),
        (13, mixed_rows(6000)),
        (2, Vec::new()),
    ] {
        with_trace_backend(log_t, 4, rows, |backend, _| {
            run_pair_on(backend, false, true);
        });
    }
}

#[cfg(feature = "field-inline")]
#[test]
fn streaming_phases_match_reference_with_field_inline_instructions() {
    run_pair_with_field_inline_program(|backend| run_pair_on(backend, false, true));
}

/// A one-cycle domain has no round: the streaming cycle phase holds its RA
/// factors dense from the start and opens the reference kernel's claims.
#[test]
fn streaming_cycle_phase_serves_a_one_cycle_domain() {
    with_trace_backend(0, 4, mixed_rows(1), |backend, _| {
        let address = AddressCase::new(backend, false);
        let case = CycleCase::new(
            backend,
            &address,
            synthetic_point(address.dimensions.log_k(), 101),
        );
        let mut session = ProofSession::default();
        let mut reference =
            <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>>::prepare(
                &ReferenceBackend,
                &mut session,
                backend,
                case.inputs(),
            )
            .unwrap();
        let mut streaming = StreamingBytecodeReadRafCycle
            .prepare(&mut session, backend, case.inputs())
            .unwrap();
        assert_eq!(
            reference.output_claims(&case.claims).unwrap(),
            streaming.output_claims(&case.claims).unwrap()
        );
    });
}

/// Over a padded trace the streaming cycle kernel holds its coefficient
/// table for the cycles before the padding only, and its dense RA rows for
/// the bound indices that hold one.
#[test]
fn streaming_cycle_phase_tables_stop_at_the_padding() {
    with_trace_backend(11, 4, mixed_rows(1500), |backend, _| {
        let address = AddressCase::new(backend, false);
        let case = CycleCase::new(
            backend,
            &address,
            synthetic_point(address.dimensions.log_k(), 101),
        );
        let mut kernel = CycleKernel::streaming(backend, &case.inputs()).unwrap();
        assert_eq!(kernel.combined.prefix_len(), 1500);
        for round in 0..4 {
            kernel.bind(fr(31 + round)).unwrap();
        }
        assert_eq!(
            kernel.ra.lazy().map(LazyFoldedRa::prefix_len),
            Some(1500usize.div_ceil(16))
        );
    });
}

#[test]
fn streaming_address_phase_rejects_a_bytecode_index_outside_the_table() {
    let stray = alu_row(5);
    let stray = JoltTraceRow::new(
        stray.instruction(),
        stray.registers(),
        stray.ram_access(),
        4,
    )
    .unwrap();
    with_trace_backend(2, 4, vec![load_row(0x8000_1000), stray], |backend, _| {
        assert_eq!(backend.program_preprocessing().bytecode.bytecode.len(), 4);
        let case = AddressCase::new(backend, false);
        assert!(matches!(
            StreamingBytecodeReadRafAddress
                .prepare(&mut ProofSession::default(), backend, case.inputs())
                .map(|_| ()),
            Err(KernelError::Witness(
                WitnessError::InvalidWitnessData { .. }
            ))
        ));
    });
}

/// `with_streaming` puts the streaming kernels in both bytecode read-RAF
/// slots. Streamed and default proofs are byte-equal, so only a kernel's
/// behavior shows which one is installed: each refuses a witness without
/// resident rows, which the default kernels collect. Dory stands in for the
/// commitment scheme; the akita build's `optimized` takes another.
#[test]
fn with_streaming_installs_both_bytecode_read_raf_kernels() {
    structured_fixture(7).with_plane(3, |resident| {
        // A configured domain shorter than the trace leaves no resident
        // rows; chunk visits still serve its `2^log_t` cycles.
        let mut chunk_only = TraceBackend::new(
            resident.config.clone(),
            JoltVmWitnessInputs::new(
                &resident.program,
                &resident.preprocessing,
                resident.trace.clone(),
            ),
        );
        chunk_only.config.log_t = 1;
        let address = AddressCase::new(&chunk_only, false);
        let cycle = CycleCase::new(
            &chunk_only,
            &address,
            synthetic_point(address.dimensions.log_k(), 101),
        );
        let prepare = |backend: JoltBackend<Fr, DoryScheme>| {
            (
                backend
                    .bytecode_read_raf_address
                    .prepare(&mut ProofSession::default(), &chunk_only, address.inputs())
                    .map(|_| ()),
                backend
                    .bytecode_read_raf_cycle
                    .prepare(&mut ProofSession::default(), &chunk_only, cycle.inputs())
                    .map(|_| ()),
            )
        };
        let default = prepare(JoltBackend::optimized());
        default.0.unwrap();
        default.1.unwrap();
        let streaming = prepare(JoltBackend::optimized().with_streaming(StreamingFamilies {
            instruction_rows: true,
            ..StreamingFamilies::default()
        }));
        assert!(matches!(
            streaming.0,
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            streaming.1,
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
    });
}
