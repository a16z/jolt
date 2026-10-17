use std::ops::Range;

use jolt_claims::protocols::jolt::geometry::dimensions::{
    ReadWriteDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
};
#[cfg(feature = "akita")]
use jolt_field::Prime128OffsetA7F7;
use jolt_field::{Fr, JoltField};
use jolt_poly::Polynomial;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_witness::{
    ChunkVisitor, JoltWitnessOracle, JoltWitnessPlane, ProgramSource, RowSource, Shape,
    TraceBackend, WitnessError,
};

use super::rows::RegisterColumns;
#[cfg(feature = "parallel")]
use super::sparse::Seeds;
use super::test_support::{
    assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture, TraceFixture,
};
use super::{OptimizedRegistersReadWrite, ReadWriteKernel, RegisterCycleRow, ROWS_LABEL};
use crate::optimized::parity::{probe_input_claim, run_lockstep, ExceptionalEq};
use crate::optimized::support::KernelRows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend, SumcheckKernel,
};

/// The trace backend without its row handle: collection keeps its seed
/// cells, and the kernel refuses it.
struct SequentialRows<'a>(&'a TraceBackend);

impl<F: JoltField> JoltWitnessOracle<F> for SequentialRows<'_> {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        JoltWitnessOracle::<F>::shape(self.0, id)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError> {
        JoltWitnessOracle::<F>::oracle_table(self.0, id)
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        JoltWitnessOracle::<F>::committed_order(self.0)
    }
}

impl RowSource for SequentialRows<'_> {
    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        self.0.visit_chunks(range, chunk_size, visitor)
    }
}

impl ProgramSource for SequentialRows<'_> {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        self.0.program_preprocessing()
    }
}

fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
    run_parity_with_phases(
        fixture,
        log_t,
        seed,
        &[(log_t, 0), (0, REGISTER_ADDRESS_BITS)],
    );
}

fn run_parity_with_phases(
    fixture: TraceFixture,
    log_t: usize,
    seed: u64,
    phase_splits: &[(usize, usize)],
) {
    run_parity_case(fixture, log_t, seed, phase_splits, None);
}

/// Every case runs on the row-handle plane in each field.
fn run_parity_case(
    fixture: TraceFixture,
    log_t: usize,
    seed: u64,
    phase_splits: &[(usize, usize)],
    exceptional: Option<ExceptionalEq>,
) {
    fixture.with_plane(log_t, |backend| {
        parity_in_field::<Fr>(backend, log_t, seed, phase_splits, exceptional);
        #[cfg(feature = "akita")]
        parity_in_field::<Prime128OffsetA7F7>(backend, log_t, seed, phase_splits, exceptional);
    });
}

fn parity_in_field<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
    seed: u64,
    phase_splits: &[(usize, usize)],
    exceptional: Option<ExceptionalEq>,
) {
    let round_challenges = challenge_sequence::<F>(log_t + REGISTER_ADDRESS_BITS, seed);
    let first_cycle = if phase_splits[0].0 == 0 {
        phase_splits[0].1
    } else {
        0
    };
    let r_cycle = exceptional.map_or_else(
        || challenge_sequence(log_t, seed ^ 0xA5A5),
        |case| case.point(log_t, round_challenges[first_cycle]),
    );
    let evaluate = |polynomial: JoltVirtualPolynomial| {
        let table =
            JoltWitnessOracle::<F>::oracle_table(witness, JoltPolynomialId::Virtual(polynomial))
                .unwrap();
        Polynomial::new(table).evaluate(&r_cycle)
    };
    let gamma = F::from_u64(0x5EED_1234_5678_9ABC);
    let claims = RegistersReadWriteInputClaims {
        rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
        rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
        rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
    };
    let points = RegistersReadWriteInputClaims {
        rd_write_value: r_cycle.clone(),
        rs1_value: r_cycle.clone(),
        rs2_value: r_cycle,
    };
    let challenges = RegistersReadWriteChallenges { gamma };
    let input_claim =
        claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
    if exceptional.is_none() {
        assert_nontrivial(input_claim);
    }
    for &(phase1, phase2) in phase_splits {
        let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
            log_t,
            REGISTER_ADDRESS_BITS,
            phase1,
            phase2,
        ));
        assert_kernel_parity(
            &OptimizedRegistersReadWrite,
            witness,
            &relation,
            &claims,
            &points,
            &challenges,
            input_claim,
            &round_challenges,
        );
    }
}

#[test]
fn parity_structured_odd_log_t() {
    run_parity(structured_fixture(8), 3, 17);
}

#[test]
fn parity_structured_even_log_t() {
    let log_t = 4;
    run_parity_with_phases(
        structured_fixture(16),
        log_t,
        23,
        &[(log_t, 0), (0, REGISTER_ADDRESS_BITS), (log_t / 2, 0)],
    );
}

#[test]
fn parity_past_lut_saturation() {
    // log_t = 6 runs three LUT-mode binds, the deref at the fourth, and
    // two more cycle binds on direct field coefficients.
    run_parity(structured_fixture(60), 6, 29);
}

#[test]
fn parity_address_first_across_chunk_boundaries_and_padding() {
    run_parity_with_phases(
        structured_fixture((1 << 13) + 17),
        14,
        37,
        &[(0, REGISTER_ADDRESS_BITS)],
    );
}

/// At log_t = 14 the cycle rounds walk 128-cycle `x_out` groups and the
/// second bind four 4096-cycle blocks; padding starts inside the third.
#[test]
fn parity_cycle_first_across_seed_blocks_and_padding() {
    run_parity_with_phases(structured_fixture((1 << 13) + 17), 14, 43, &[(14, 0)]);
}

#[test]
fn parity_minimal_padded_trace() {
    let mut fixture = TraceFixture::new();
    fixture.op(Some(3), Some(1), Some(2));
    fixture.op(Some(3), Some(3), None);
    fixture.op(None, Some(3), Some(3));
    run_parity(fixture, 2, 31);
}

#[test]
fn parity_single_cycle_round() {
    let mut fixture = TraceFixture::new();
    fixture.op(Some(9), Some(9), Some(9));
    fixture.op(Some(9), None, Some(9));
    run_parity(fixture, 1, 41);
}

#[test]
fn parity_exceptional_eq_across_sparse_layouts_and_orders() {
    for log_t in [3usize, 4, 14] {
        for phase in [(log_t, 0), (0, REGISTER_ADDRESS_BITS)] {
            for case in ExceptionalEq::ALL {
                run_parity_case(
                    structured_fixture(1 << log_t),
                    log_t,
                    79,
                    &[phase],
                    Some(case),
                );
            }
        }
    }
}

/// Pins which seed source the parity runs exercise: only a plane without
/// a row handle keeps the seed cells in memory.
#[cfg(feature = "parallel")]
#[test]
fn seed_cells_are_collected_only_without_trace_rows() {
    structured_fixture(8).with_plane(3, |backend| {
        let (_, seeds) = RegisterColumns::collect::<Fr>(backend, 8).unwrap();
        assert!(matches!(seeds, Seeds::Rows(_)));
        let (_, seeds) = RegisterColumns::collect::<Fr>(&SequentialRows(backend), 8).unwrap();
        assert!(matches!(seeds, Seeds::Collected(cells) if !cells.is_empty()));
    });
}

#[test]
fn collection_rejects_out_of_domain_registers_with_either_row_source() {
    let mut fixture = structured_fixture(5);
    fixture.unchecked_read(200);
    fixture.with_plane(3, |backend| {
        let sequential = SequentialRows(backend);
        for witness in [backend as &dyn JoltWitnessPlane<Fr>, &sequential] {
            assert!(matches!(
                RegisterColumns::collect(witness, 8),
                Err(KernelError::Witness(WitnessError::InvalidWitnessData { reason, .. }))
                    if reason.contains("register index 200")
            ));
        }
    });
}

/// [`structured_fixture`] ending with x0 as `rd`, `rs1` and `rs2` of one
/// cycle, then read back.
fn x0_fixture(cycles: usize) -> TraceFixture {
    let mut fixture = structured_fixture(cycles - 2);
    fixture.op(Some(0), Some(0), Some(0));
    fixture.op(Some(9), Some(0), None);
    fixture
}

/// The reference kernel over the resident plane against the optimized
/// kernel whose streamed rounds and operand walk read `rows`, in lockstep
/// through every round: equal round polynomials, `EqCycle` pins and output
/// claims.
fn assert_rows_parity<F: JoltField>(
    resident: &TraceBackend,
    rows: KernelRows<RegisterCycleRow>,
    log_t: usize,
    seed: u64,
) {
    let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
        log_t,
        REGISTER_ADDRESS_BITS,
        log_t,
        0,
    ));
    let r_cycle: Vec<F> = challenge_sequence(log_t, seed ^ 0xA5A5);
    let claims = RegistersReadWriteInputClaims {
        rd_write_value: F::zero(),
        rs1_value: F::zero(),
        rs2_value: F::zero(),
    };
    let points = RegistersReadWriteInputClaims {
        rd_write_value: r_cycle.clone(),
        rs1_value: r_cycle.clone(),
        rs2_value: r_cycle,
    };
    let challenges = RegistersReadWriteChallenges {
        gamma: F::from_u64(0x5EED_1234_5678_9ABC),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };
    let mut reference = ReferenceBackend
        .prepare(&mut ProofSession::default(), resident, inputs())
        .unwrap();
    let mut optimized = ReadWriteKernel::new(resident, rows, &inputs()).unwrap();
    let input_claim = probe_input_claim(reference.as_mut());
    assert_nontrivial(input_claim);
    let round_challenges = challenge_sequence::<F>(log_t + REGISTER_ADDRESS_BITS, seed);
    run_lockstep(
        reference.as_mut(),
        &mut optimized,
        input_claim,
        &round_challenges,
    );
    let output_points = relation
        .derive_opening_points(&round_challenges, &points)
        .unwrap();
    reference
        .validate_derived_tables(&relation, &points, &output_points, &challenges)
        .unwrap();
    optimized
        .validate_derived_tables(&relation, &points, &output_points, &challenges)
        .unwrap();
    assert_eq!(
        optimized.output_claims(&claims).unwrap(),
        reference.output_claims(&claims).unwrap(),
        "output claims, log_t {log_t}"
    );
}

/// Rounds 0 and 1 and the operand walk read the witness's trace rows, with
/// task boundaries and a padded proof suffix inside the domain, while the
/// second bind collects from the same plane. At log_t 13 every walk spans
/// several tasks.
#[test]
fn streamed_passes_over_trace_rows_match_reference() {
    fn case<F: JoltField>(resident: &TraceBackend, log_t: usize) {
        let rows = KernelRows::from_witness::<F>(resident, log_t, ROWS_LABEL).unwrap();
        assert_rows_parity::<F>(resident, rows, log_t, 83);
    }
    for (log_t, cycles) in [(1, 2), (3, 7), (6, 63)] {
        x0_fixture(cycles).with_plane(log_t, |resident| {
            case::<Fr>(resident, log_t);
            #[cfg(feature = "akita")]
            case::<Prime128OffsetA7F7>(resident, log_t);
        });
    }
    x0_fixture((1 << 13) - 3).with_plane(13, |resident| case::<Fr>(resident, 13));
}

fn prepare_on<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
) -> Result<(), KernelError<F>> {
    let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
        log_t,
        REGISTER_ADDRESS_BITS,
        log_t,
        0,
    ));
    let r_cycle: Vec<F> = challenge_sequence(log_t, 3);
    let claims = RegistersReadWriteInputClaims {
        rd_write_value: F::zero(),
        rs1_value: F::zero(),
        rs2_value: F::zero(),
    };
    let points = RegistersReadWriteInputClaims {
        rd_write_value: r_cycle.clone(),
        rs1_value: r_cycle.clone(),
        rs2_value: r_cycle,
    };
    let challenges = RegistersReadWriteChallenges {
        gamma: F::from_u64(5),
    };
    OptimizedRegistersReadWrite
        .prepare(
            &mut ProofSession::default(),
            witness,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            },
        )
        .map(drop)
}

/// A witness without a row handle, or with rows over another cycle domain,
/// fails typed before anything is collected.
#[test]
fn preparation_refuses_missing_and_foreign_rows() {
    structured_fixture(7).with_plane(3, |backend| {
        assert!(matches!(
            prepare_on::<Fr>(&SequentialRows(backend), 3),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            prepare_on::<Fr>(backend, 4),
            Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
        ));
    });
}
