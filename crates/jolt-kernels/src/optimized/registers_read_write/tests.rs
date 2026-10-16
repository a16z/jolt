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
use super::OptimizedRegistersReadWrite;
use crate::optimized::parity::ExceptionalEq;
use crate::KernelError;

/// The trace backend without its random access, so the kernel collects seed
/// cells instead of regenerating them from rows.
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

/// Every case runs on the random-access plane (seed cells regenerated from
/// rows) and on [`SequentialRows`] (seed cells collected), in each field.
fn run_parity_case(
    fixture: TraceFixture,
    log_t: usize,
    seed: u64,
    phase_splits: &[(usize, usize)],
    exceptional: Option<ExceptionalEq>,
) {
    fixture.with_plane(log_t, |backend| {
        let sequential = SequentialRows(backend);
        parity_in_field::<Fr>(
            [backend, &sequential],
            log_t,
            seed,
            phase_splits,
            exceptional,
        );
        #[cfg(feature = "akita")]
        parity_in_field::<Prime128OffsetA7F7>(
            [backend, &sequential],
            log_t,
            seed,
            phase_splits,
            exceptional,
        );
    });
}

fn parity_in_field<F: JoltField>(
    witnesses: [&dyn JoltWitnessPlane<F>; 2],
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
        let table = JoltWitnessOracle::<F>::oracle_table(
            witnesses[0],
            JoltPolynomialId::Virtual(polynomial),
        )
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
    for witness in witnesses {
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
/// random access keeps the seed cells in memory.
#[cfg(feature = "parallel")]
#[test]
fn seed_cells_are_collected_only_without_random_access() {
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
