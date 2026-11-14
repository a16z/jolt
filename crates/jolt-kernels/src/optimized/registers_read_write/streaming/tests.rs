#[cfg(feature = "akita")]
use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial, ReadWriteDimensions};
use jolt_field::{Fr, Ring};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
};
use jolt_witness::{JoltWitnessOracle, TraceBackend};

use super::*;
use crate::optimized::parity::{probe_input_claim, ExceptionalEq};
use crate::optimized::registers_read_write::test_support::{
    challenge_sequence, structured_fixture, TraceFixture,
};
use crate::ReferenceBackend;

const ROWS: &str = "register read/write trace rows";
const DIVERGED: &str = "register frontier fill (entry counts disagree)";

/// [`structured_fixture`] ending with x0 as `rd`, `rs1` and `rs2` of one
/// cycle, then read back.
fn x0_fixture(cycles: usize) -> TraceFixture {
    let mut fixture = structured_fixture(cycles - 2);
    fixture.op(Some(0), Some(0), Some(0));
    fixture.op(Some(9), Some(0), None);
    fixture
}

/// Each level's entries by definition: the distinct `(register, t >> level)`
/// among the trace's register accesses, read off its one-hot `rs1`, `rs2`
/// and `rd` oracle tables.
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
    (0..log_t)
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

/// The cap that first admits round `depth`'s level: its structural entries
/// and increments at their actual sizes, which every earlier level exceeds.
fn cap_at_depth<F: JoltField>(structure: &[usize], depth: usize) -> usize {
    StreamedCycleState::<F>::level_bytes(structure[depth], (1 << structure.len()) >> depth)
}

/// Inputs over `log_t` cycle variables bound cycle-first, at the cycle point
/// `r_cycle`, with zero input claims.
fn with_inputs<F: JoltField, T>(
    log_t: usize,
    r_cycle: Vec<F>,
    f: impl FnOnce(ProverInputs<'_, F, RegistersReadWriteChecking<F>>) -> T,
) -> T {
    let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
        log_t,
        REGISTER_ADDRESS_BITS,
        log_t,
        0,
    ));
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
    f(ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    })
}

fn trace_rows<F: JoltField>(
    resident: &TraceBackend,
    log_t: usize,
) -> BundleStore<RegisterCycleRow> {
    BundleStore::resident::<F>(resident, log_t, ROWS).unwrap()
}

/// A streamed source's cycle state and rows.
fn streamed<F: JoltField>(
    kernel: &mut ReadWriteKernel<F>,
) -> Option<(&StreamedCycleState<F>, &mut BundleStore<RegisterCycleRow>)> {
    match &mut kernel.cycle {
        CycleSource::Streamed { state, rows } => Some((state, rows)),
        CycleSource::Collected { .. } => None,
    }
}

/// The reference kernel over the resident plane against the kernel streaming
/// its rows with frontier cap `cap`, in lockstep through every round:
/// equal round polynomials, `EqCycle` pins and output claims. The cycle point
/// is pseudo-random or `exceptional`, and the round challenges start with
/// `leading`. Each streamed round counts the next level's entries by
/// definition; the filled level holds its entries by definition in exactly
/// their capacity, with one increment per row, until the final cycle bind
/// releases it. Returns the filling round.
fn assert_rows_parity<F: JoltField>(
    resident: &TraceBackend,
    log_t: usize,
    seed: u64,
    leading: &[F],
    exceptional: Option<ExceptionalEq>,
    cap: usize,
) -> usize {
    let mut round_challenges = challenge_sequence::<F>(log_t + REGISTER_ADDRESS_BITS, seed);
    round_challenges[..leading.len()].copy_from_slice(leading);
    let r_cycle = exceptional.map_or_else(
        || challenge_sequence(log_t, seed ^ 0xA5A5),
        |case| case.point(log_t, round_challenges[0]),
    );
    let structure = structural_counts::<F>(resident, log_t);
    with_inputs(log_t, r_cycle, |inputs| {
        let reference_inputs = ProverInputs {
            relation: inputs.relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), resident, reference_inputs)
            .unwrap();
        let mut streaming =
            ReadWriteKernel::streamed(trace_rows::<F>(resident, log_t), &inputs, cap).unwrap();
        let input_claim = probe_input_claim(reference.as_mut());
        if exceptional.is_none() {
            assert!(
                input_claim != F::zero() && input_claim != F::one(),
                "degenerate fixture"
            );
        }
        let mut claim = input_claim;
        let mut filled = None;
        for (round, &challenge) in round_challenges.iter().enumerate() {
            let bind = round
                .checked_sub(1)
                .map(|previous| round_challenges[previous]);
            let expected = reference.prove_round(bind, round, claim).unwrap();
            let actual = streaming.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                expected.coefficients(),
                actual.coefficients(),
                "round {round}, cap {cap}"
            );
            let (cycle, _) = streamed(&mut streaming).unwrap();
            if round < log_t {
                match cycle.filled() {
                    Some(shape) => {
                        let fill_round = *filled.get_or_insert(round);
                        assert_eq!(
                            shape,
                            (
                                structure[round],
                                structure[fill_round],
                                (1 << log_t) >> round
                            ),
                            "round {round}, cap {cap}"
                        );
                    }
                    None => assert_eq!(
                        cycle.counted(),
                        structure.get(round + 1).copied(),
                        "round {round}, cap {cap}"
                    ),
                }
            } else {
                assert_eq!(cycle.filled(), Some((0, 0, 0)), "round {round}");
            }
            claim = expected.evaluate(challenge);
        }
        let last = round_challenges[round_challenges.len() - 1];
        reference.finish_rounds(last).unwrap();
        streaming.finish_rounds(last).unwrap();
        let output_points = inputs
            .relation
            .derive_opening_points(&round_challenges, inputs.points)
            .unwrap();
        streaming
            .validate_derived_tables(
                inputs.relation,
                inputs.points,
                &output_points,
                inputs.challenges,
            )
            .unwrap();
        assert_eq!(
            streaming.output_claims(inputs.claims).unwrap(),
            reference.output_claims(inputs.claims).unwrap(),
            "output claims, log_t {log_t}, cap {cap}"
        );
        filled.unwrap()
    })
}

/// Every round and the operand walk read the witness's resident rows across
/// several tasks, with a padded proof suffix inside the domain: the
/// production cap fills round 1, and a cap no earlier level meets fills the
/// last cycle round, whose one pair spans two 4096-row visit callbacks.
#[test]
fn streamed_passes_over_trace_rows_match_reference() {
    x0_fixture((1 << 13) - 3).with_plane(13, |resident| {
        let parity = |cap| assert_rows_parity::<Fr>(resident, 13, 83, &[], None, cap);
        assert_eq!(parity(DENSE_TABLE_CAP), 1);
        assert_eq!(parity(0), 12);
    });
}

/// Caps that fill the frontier at each LUT level (1 to 3), at the first field
/// levels (4 and 6) and at the last cycle round, which fills whatever the
/// cap; a cap one byte below round 2's level fills round 3.
#[test]
fn streamed_frontier_matches_reference_at_every_crossover() {
    fn case<F: JoltField>(resident: &TraceBackend) {
        let structure = structural_counts::<F>(resident, 8);
        let parity = |cap| assert_rows_parity::<F>(resident, 8, 97, &[], None, cap);
        for depth in [1, 2, 3, 4, 6, 7] {
            assert_eq!(
                parity(cap_at_depth::<F>(&structure, depth)),
                depth,
                "depth {depth}"
            );
        }
        assert_eq!(parity(cap_at_depth::<F>(&structure, 2) - 1), 3);
        assert_eq!(parity(0), 7);
    }
    x0_fixture(253).with_plane(8, |resident| {
        case::<Fr>(resident);
        #[cfg(feature = "akita")]
        case::<AkitaField>(resident);
    });
}

/// Literal 0 and 1 challenges at each of the first five binds, at random,
/// zero, one and zero-prefix cycle points, filling at rounds 1, 2, 4 and 5:
/// the coefficient binds' 0/1 shortcuts, the streamed and filled `q(1)`
/// passes, counting under a zero Gruen scalar, and filled entries whose
/// coefficients vanish but which the structure keeps.
#[test]
fn zero_one_challenges_and_degenerate_points_keep_structural_entries() {
    fn case<F: JoltField>(resident: &TraceBackend) {
        let structure = structural_counts::<F>(resident, 6);
        let (zero, one) = (F::zero(), F::one());
        for depth in [1, 2, 4, 5] {
            for leading in [[zero, one, zero, one, one], [one, zero, one, zero, zero]] {
                for exceptional in std::iter::once(None).chain(ExceptionalEq::ALL.map(Some)) {
                    let filled = assert_rows_parity::<F>(
                        resident,
                        6,
                        89,
                        &leading,
                        exceptional,
                        cap_at_depth::<F>(&structure, depth),
                    );
                    assert_eq!(filled, depth);
                }
            }
        }
    }
    structured_fixture(60).with_plane(6, |resident| {
        case::<Fr>(resident);
        #[cfg(feature = "akita")]
        case::<AkitaField>(resident);
    });
}

/// An out-of-domain register fails round 0's count, and rows with fewer or
/// more entries than the last pass counted fail the filling round, each
/// naming its cause, before any entry is published; a cycle domain past
/// `u32` row indices is refused.
#[test]
fn row_failures_and_count_divergence_are_typed_and_publish_no_entries() {
    let challenges = challenge_sequence::<Fr>(3 + REGISTER_ADDRESS_BITS, 11);
    let new_kernel = |witness: &TraceBackend, cap| {
        with_inputs(3, challenge_sequence(3, 3), |inputs| {
            ReadWriteKernel::streamed(trace_rows::<Fr>(witness, 3), &inputs, cap)
        })
        .unwrap()
    };
    let prove = |kernel: &mut ReadWriteKernel<Fr>, round: usize| {
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        kernel.prove_round(bind, round, Fr::from_u64(0))
    };
    let failure = |result: Result<_, SumcheckError<Fr>>| match result {
        Err(SumcheckError::MissingEvaluationSource { kind }) => Some(kind),
        _ => None,
    };
    // `fewer` repeats `resident`'s first four cycles and leaves the next four
    // empty, so filling either one's rows against the other's count
    // diverges.
    let mut foreign = structured_fixture(5);
    foreign.unchecked_read(200);
    structured_fixture(7).with_plane(3, |resident| {
        structured_fixture(4).with_plane(3, |fewer| {
            foreign.with_plane(3, |foreign| {
                let mut kernel = new_kernel(foreign, DENSE_TABLE_CAP);
                assert_eq!(failure(prove(&mut kernel, 0)), Some(ROWS));
                assert_eq!(streamed(&mut kernel).unwrap().0.filled(), None);
            });
            for (counted, filled) in [(resident, fewer), (fewer, resident)] {
                let structure = structural_counts::<Fr>(counted, 3);
                for depth in [1, 2] {
                    let mut kernel = new_kernel(counted, cap_at_depth::<Fr>(&structure, depth));
                    for round in 0..depth {
                        let _ = prove(&mut kernel, round).unwrap();
                    }
                    *streamed(&mut kernel).unwrap().1 = trace_rows::<Fr>(filled, 3);
                    assert_eq!(
                        failure(prove(&mut kernel, depth)),
                        Some(DIVERGED),
                        "depth {depth}"
                    );
                    assert_eq!(streamed(&mut kernel).unwrap().0.filled(), None);
                }
            }
            let wide = with_inputs(33, challenge_sequence::<Fr>(33, 3), |inputs| {
                ReadWriteKernel::streamed(trace_rows::<Fr>(resident, 3), &inputs, 0).map(drop)
            });
            assert!(matches!(wide, Err(KernelError::InvariantViolation { .. })));
        });
    });
}
