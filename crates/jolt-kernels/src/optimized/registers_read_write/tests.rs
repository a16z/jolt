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
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_witness::{
    ChunkVisitor, JoltWitnessOracle, JoltWitnessPlane, ProgramSource, RowSource, Shape,
    TraceBackend, WitnessError,
};

use super::sparse::layout::SparseEntry;
use super::sparse::{CyclePhase, Frontier, NextLevel};
use super::test_support::{
    assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture, TraceFixture,
};
use super::{
    OptimizedRegistersReadWrite, ReadWriteKernel, RegisterCycleRow, REGISTER_FRONTIER_CAP,
    ROWS_LABEL,
};
use crate::optimized::parity::{probe_input_claim, replayed, ExceptionalEq};
use crate::optimized::support::KernelRows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend, SumcheckKernel,
    SumcheckKernelError,
};

/// The trace backend without its row handle, which the kernel refuses.
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
fn parity_in_place_binds_after_fill() {
    // log_t = 6: round 1's pass fills the field entries, then four cycle
    // binds compact the entries and the increment in place.
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

/// At log_t = 14 the cycle rounds walk 128-cycle `x_out` groups in several
/// tasks, so round 1 fills each task's counted slots; padding starts inside
/// the domain.
#[test]
fn parity_cycle_first_across_tasks_and_padding() {
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

/// [`structured_fixture`] ending with x0 as `rd`, `rs1` and `rs2` of one
/// cycle, then read back.
fn x0_fixture(cycles: usize) -> TraceFixture {
    let mut fixture = structured_fixture(cycles - 2);
    fixture.op(Some(0), Some(0), Some(0));
    fixture.op(Some(9), Some(0), None);
    fixture
}

/// Each level's entries by definition: the distinct `(register, t >> level)`
/// among the resident trace's register accesses, read off its one-hot `rs1`,
/// `rs2` and `rd` oracle tables.
fn structural_counts<F: JoltField>(resident: &TraceBackend, log_t: usize) -> Vec<usize> {
    let mut accesses = Vec::new();
    for polynomial in [
        JoltVirtualPolynomial::Rs1Ra,
        JoltVirtualPolynomial::Rs2Ra,
        JoltVirtualPolynomial::RdWa,
    ] {
        let table =
            JoltWitnessOracle::<F>::oracle_table(resident, JoltPolynomialId::Virtual(polynomial))
                .unwrap();
        for (index, value) in table.iter().enumerate() {
            if *value != F::zero() {
                accesses.push((index >> log_t, index & ((1 << log_t) - 1)));
            }
        }
    }
    (0..=log_t)
        .map(|level| {
            let mut entries: Vec<_> = accesses
                .iter()
                .map(|&(register, t)| (register, t >> level))
                .collect();
            entries.sort_unstable();
            entries.dedup();
            entries.len()
        })
        .collect()
}

/// The frontier cap that first admits round `depth`'s level: its structural
/// entries and increments at their actual sizes, which every earlier level
/// exceeds. Round `log_t` stands for never.
fn cap_at_depth<F: JoltField>(structure: &[usize], log_t: usize, depth: usize) -> usize {
    if depth == log_t {
        return 0;
    }
    size_of::<SparseEntry<F, F>>() * structure[depth] + size_of::<F>() * ((1 << log_t) >> depth)
}

/// The reference kernel over the resident plane against the optimized
/// kernel over `rows` with frontier cap `cap`, in lockstep through every
/// round: equal round polynomials, `EqCycle` pins and output claims. The
/// cycle point is pseudo-random or `exceptional`, and the round challenges
/// start with `leading`. Each streamed round counts the next level's entries
/// by definition, and the last keeps its pair; the filled frontier holds
/// each level's entries in its filling capacity until the final cycle bind
/// releases it. Returns the filling round.
fn assert_rows_parity<F: JoltField>(
    resident: &TraceBackend,
    rows: KernelRows<RegisterCycleRow>,
    log_t: usize,
    seed: u64,
    leading: &[F],
    exceptional: Option<ExceptionalEq>,
    cap: usize,
) -> Option<usize> {
    let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
        log_t,
        REGISTER_ADDRESS_BITS,
        log_t,
        0,
    ));
    let mut round_challenges = challenge_sequence::<F>(log_t + REGISTER_ADDRESS_BITS, seed);
    round_challenges[..leading.len()].copy_from_slice(leading);
    let r_cycle = exceptional.map_or_else(
        || challenge_sequence(log_t, seed ^ 0xA5A5),
        |case| case.point(log_t, round_challenges[0]),
    );
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
    let mut optimized = ReadWriteKernel::new(rows, &inputs(), cap).unwrap();
    let input_claim = probe_input_claim(reference.as_mut());
    if exceptional.is_none() {
        assert_nontrivial(input_claim);
    }
    let structure = structural_counts::<F>(resident, log_t);
    let mut claim = input_claim;
    let mut filled = None;
    for (round, &challenge) in round_challenges.iter().enumerate() {
        let bind = round
            .checked_sub(1)
            .map(|previous| round_challenges[previous]);
        let expected = reference.prove_round(bind, round, claim).unwrap();
        let actual = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            expected.coefficients(),
            actual.coefficients(),
            "round {round}, cap {cap}"
        );
        match &optimized.cycle.0 {
            CyclePhase::Streamed(Frontier {
                next: Some(NextLevel::Counts(counts)),
                ..
            }) => assert_eq!(
                counts.iter().sum::<usize>(),
                structure[round + 1],
                "round {round}"
            ),
            CyclePhase::Streamed(Frontier {
                next: Some(NextLevel::Final(_)),
                ..
            }) => assert_eq!(round + 1, log_t),
            CyclePhase::Direct { entries, inc } if round < log_t => {
                let (_, capacity) = *filled.get_or_insert((round, structure[round]));
                assert_eq!(
                    (entries.len(), entries.capacity(), inc.len()),
                    (structure[round], capacity, (1 << log_t) >> round),
                    "round {round}, cap {cap}"
                );
            }
            CyclePhase::Direct { entries, inc } => {
                assert_eq!((entries.capacity(), inc.len()), (0, 0), "round {round}");
            }
            CyclePhase::Streamed(_) => unreachable!("round {round} left no pass result"),
        }
        claim = expected.evaluate(challenge);
    }
    let last = round_challenges[round_challenges.len() - 1];
    reference.finish_rounds(last).unwrap();
    optimized.finish_rounds(last).unwrap();
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
        "output claims, log_t {log_t}, cap {cap}"
    );
    filled.map(|(round, _)| round)
}

/// Every round and the operand walk read rows replayed from two-row
/// checkpoints, with task boundaries and a padded proof suffix inside the
/// domain; the production cap fills round 1. At log_t 13 every walk spans
/// several tasks, and filling at round 12 or never, the last rounds' pairs
/// span two 4096-row visit callbacks.
#[test]
fn streamed_passes_over_replayed_rows_match_reference() {
    fn case<F: JoltField>(resident: &TraceBackend, log_t: usize, cap: usize) -> Option<usize> {
        let rows =
            KernelRows::from_witness::<F>(&replayed(resident, 2, None), log_t, ROWS_LABEL).unwrap();
        assert_rows_parity::<F>(resident, rows, log_t, 83, &[], None, cap)
    }
    for (log_t, cycles) in [(1, 2), (3, 7), (6, 63)] {
        x0_fixture(cycles).with_plane(log_t, |resident| {
            let filled = (log_t > 1).then_some(1);
            assert_eq!(case::<Fr>(resident, log_t, REGISTER_FRONTIER_CAP), filled);
            #[cfg(feature = "akita")]
            assert_eq!(
                case::<Prime128OffsetA7F7>(resident, log_t, REGISTER_FRONTIER_CAP),
                filled
            );
        });
    }
    x0_fixture((1 << 13) - 3).with_plane(13, |resident| {
        let structure = structural_counts::<Fr>(resident, 13);
        assert_eq!(case::<Fr>(resident, 13, REGISTER_FRONTIER_CAP), Some(1));
        for depth in [12, 13] {
            let filled = case::<Fr>(resident, 13, cap_at_depth::<Fr>(&structure, 13, depth));
            assert_eq!(filled, (depth < 13).then_some(depth));
        }
    });
}

/// Caps that fill the frontier at each LUT level (1 to 3), at the first field
/// levels (4 and 6) or never, over resident and replayed rows; a cap one byte
/// below round 2's level fills round 3.
#[test]
fn streamed_frontier_matches_reference_at_every_crossover() {
    fn case<F: JoltField>(resident: &TraceBackend) {
        let structure = structural_counts::<F>(resident, 8);
        let replay = replayed(resident, 2, None);
        for witness in [resident, &replay] {
            let parity = |cap| {
                let rows = KernelRows::from_witness::<F>(witness, 8, ROWS_LABEL).unwrap();
                assert_rows_parity::<F>(resident, rows, 8, 97, &[], None, cap)
            };
            for depth in [1, 2, 3, 4, 6, 8] {
                let filled = parity(cap_at_depth::<F>(&structure, 8, depth));
                assert_eq!(filled, (depth < 8).then_some(depth), "depth {depth}");
            }
            assert_eq!(parity(cap_at_depth::<F>(&structure, 8, 2) - 1), Some(3));
        }
    }
    x0_fixture(253).with_plane(8, |resident| {
        case::<Fr>(resident);
        #[cfg(feature = "akita")]
        case::<Prime128OffsetA7F7>(resident);
    });
}

/// Literal 0 and 1 challenges at each of the first five binds, at random,
/// zero, one and zero-prefix cycle points, filling at rounds 1, 2 and 4 or
/// never: the coefficient binds' 0/1 shortcuts, the streamed and filled
/// `q(1)` passes, counting under a zero Gruen scalar, and filled entries
/// whose coefficients vanish but which the structure keeps.
#[test]
fn zero_one_challenges_and_degenerate_points_keep_structural_entries() {
    fn case<F: JoltField>(resident: &TraceBackend) {
        let structure = structural_counts::<F>(resident, 6);
        let (zero, one) = (F::zero(), F::one());
        for depth in [1, 2, 4, 6] {
            for leading in [[zero, one, zero, one, one], [one, zero, one, zero, zero]] {
                for exceptional in std::iter::once(None).chain(ExceptionalEq::ALL.map(Some)) {
                    let rows = KernelRows::from_witness::<F>(resident, 6, ROWS_LABEL).unwrap();
                    let cap = cap_at_depth::<F>(&structure, 6, depth);
                    let filled =
                        assert_rows_parity::<F>(resident, rows, 6, 89, &leading, exceptional, cap);
                    assert_eq!(filled, (depth < 6).then_some(depth));
                }
            }
        }
    }
    structured_fixture(60).with_plane(6, |resident| {
        case::<Fr>(resident);
        #[cfg(feature = "akita")]
        case::<Prime128OffsetA7F7>(resident);
    });
}

/// Calls `f` on inputs over `log_t` cycle variables bound in `(phase1,
/// phase2)` rounds, at a pseudo-random cycle point with zero input claims.
fn with_inputs<F: JoltField, T>(
    log_t: usize,
    (phase1, phase2): (usize, usize),
    f: impl FnOnce(ProverInputs<'_, F, RegistersReadWriteChecking<F>>) -> T,
) -> T {
    let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
        log_t,
        REGISTER_ADDRESS_BITS,
        phase1,
        phase2,
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
    f(ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    })
}

fn prepare_on<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
    phases: (usize, usize),
) -> Result<(), KernelError<F>> {
    with_inputs(log_t, phases, |inputs| {
        OptimizedRegistersReadWrite
            .prepare(&mut ProofSession::default(), witness, inputs)
            .map(drop)
    })
}

/// A witness without a row handle, rows over another cycle domain, replayed
/// rows bound address-first, or a cycle domain past `u32` row indices fail
/// typed; replayed rows bound cycle-first prepare.
#[test]
fn preparation_refuses_missing_foreign_and_address_first_replayed_rows() {
    structured_fixture(7).with_plane(3, |backend| {
        let replay = replayed(backend, 2, None);
        assert!(matches!(
            prepare_on::<Fr>(&SequentialRows(backend), 3, (3, 0)),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            prepare_on::<Fr>(backend, 4, (4, 0)),
            Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
        ));
        assert!(prepare_on::<Fr>(&replay, 3, (3, 0)).is_ok());
        assert!(matches!(
            prepare_on::<Fr>(&replay, 3, (0, REGISTER_ADDRESS_BITS)),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        let rows = KernelRows::from_witness::<Fr>(&replay, 3, ROWS_LABEL).unwrap();
        let wide = with_inputs::<Fr, _>(33, (33, 0), |inputs| {
            ReadWriteKernel::new(rows, &inputs, REGISTER_FRONTIER_CAP)
        });
        assert!(matches!(wide, Err(KernelError::InvariantViolation { .. })));
    });
}

/// A replay failure or an out-of-domain register fails round 0's count, and
/// a replay failure fails a later count; a replay failure in the filling
/// round, or rows with fewer or more entries than the last pass counted,
/// fails it before any entry is published, filling at round 1 or 2; a replay
/// failure after a never-filled run fails the operand walk.
#[test]
fn row_failures_and_count_divergence_publish_no_entries() {
    fn case<F: JoltField>(resident: &TraceBackend, fewer: &TraceBackend, foreign: &TraceBackend) {
        let rows =
            |witness: &TraceBackend| KernelRows::from_witness::<F>(witness, 3, ROWS_LABEL).unwrap();
        let failing = || rows(&replayed(resident, 2, Some(5)));
        let new_kernel = |source, cap| {
            with_inputs::<F, _>(3, (3, 0), |inputs| {
                ReadWriteKernel::new(source, &inputs, cap)
            })
            .unwrap()
        };
        let challenges = challenge_sequence::<F>(3 + REGISTER_ADDRESS_BITS, 11);
        let prove = |kernel: &mut ReadWriteKernel<F>, round: usize| {
            let bind = round.checked_sub(1).map(|previous| challenges[previous]);
            kernel.prove_round(bind, round, F::zero())
        };
        let missing = |result: Result<_, SumcheckError<F>>| {
            matches!(result, Err(SumcheckError::MissingEvaluationSource { .. }))
        };
        for source in [failing(), rows(foreign)] {
            let mut kernel = new_kernel(source, REGISTER_FRONTIER_CAP);
            assert!(missing(prove(&mut kernel, 0)));
            assert!(matches!(kernel.cycle.0, CyclePhase::Streamed(_)));
        }
        let mut kernel = new_kernel(rows(resident), 0);
        let _ = prove(&mut kernel, 0).unwrap();
        kernel.rows = failing();
        assert!(missing(prove(&mut kernel, 1)));
        assert!(matches!(kernel.cycle.0, CyclePhase::Streamed(_)));
        for (counted, filled) in [
            (resident, None),
            (resident, Some(fewer)),
            (fewer, Some(resident)),
        ] {
            let structure = structural_counts::<F>(counted, 3);
            for depth in [1, 2] {
                let cap = cap_at_depth::<F>(&structure, 3, depth);
                let mut kernel = new_kernel(rows(counted), cap);
                for round in 0..depth {
                    let _ = prove(&mut kernel, round).unwrap();
                }
                kernel.rows = filled.map_or_else(failing, rows);
                assert!(missing(prove(&mut kernel, depth)), "depth {depth}");
                assert!(matches!(kernel.cycle.0, CyclePhase::Streamed(_)));
            }
        }

        let mut kernel = new_kernel(rows(resident), 0);
        let mut claim = with_inputs::<F, _>(3, (3, 0), |inputs| {
            let mut reference = ReferenceBackend
                .prepare(&mut ProofSession::default(), resident, inputs)
                .unwrap();
            probe_input_claim(reference.as_mut())
        });
        for (round, &challenge) in challenges.iter().enumerate() {
            let bind = round.checked_sub(1).map(|previous| challenges[previous]);
            claim = kernel
                .prove_round(bind, round, claim)
                .unwrap()
                .evaluate(challenge);
        }
        let last = challenges[challenges.len() - 1];
        kernel.finish_rounds(last).unwrap();
        kernel.rows = failing();
        assert!(matches!(
            with_inputs::<F, _>(3, (3, 0), |inputs| kernel.output_claims(inputs.claims)),
            Err(SumcheckKernelError::InvariantViolation { .. })
        ));
    }
    // `fewer` repeats `resident`'s first four cycles and leaves the next four
    // empty, so filling either one's rows against the other's count
    // diverges.
    let mut foreign = structured_fixture(5);
    foreign.unchecked_read(200);
    structured_fixture(7).with_plane(3, |resident| {
        structured_fixture(4).with_plane(3, |fewer| {
            foreign.with_plane(3, |foreign| case::<Fr>(resident, fewer, foreign));
        });
    });
}
