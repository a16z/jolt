use std::collections::BTreeMap;

#[cfg(feature = "akita")]
use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::registers::{
    rd_inc_val_evaluation, rd_wa_val_evaluation,
};
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_claims::NoChallenges;
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, LtPolynomial, Polynomial};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluationInputClaims;
use jolt_witness::witnesses::ToField;
use jolt_witness::{FixedBackend, JoltVmWitnessInputs, PolynomialEncoding, Shape, TraceBackend};

use super::*;
use crate::optimized::parity::probe_input_claim;
use crate::optimized::registers_read_write::test_support::{
    challenge_sequence, structured_fixture,
};
use crate::optimized::testing::trimmed;
use crate::reference::views::address_fold;
use crate::{NaiveSumcheckProver, ReferenceBackend};

type Reference<F> = dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>;

/// One relation instance over `2^log_t` cycles at `point`, with a zero input
/// claim (the parity runs probe the true one).
struct Instance<F: JoltField> {
    relation: RegistersValEvaluation<F>,
    claims: RegistersValEvaluationInputClaims<F>,
    points: RegistersValEvaluationInputClaims<Vec<F>>,
    challenges: ConcreteSumcheckChallenges<F, RegistersValEvaluation<F>>,
}

impl<F: JoltField> Instance<F> {
    fn new(log_t: usize, point: Vec<F>) -> Self {
        Self {
            relation: RegistersValEvaluation::new(TraceDimensions::new(log_t)),
            claims: RegistersValEvaluationInputClaims {
                registers_val: F::zero(),
            },
            points: RegistersValEvaluationInputClaims {
                registers_val: point,
            },
            challenges: NoChallenges::default(),
        }
    }

    fn inputs(&self) -> ProverInputs<'_, F, RegistersValEvaluation<F>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// Rows over the whole signed increment range (both `±(2^64 − 1)` extremes of
/// a u64 write delta and the `i128` limits) and both ends of the register
/// domain, with canonical absence: a cycle without rd carries increment zero.
fn synthetic_rows(log_t: usize, seed: u64) -> Vec<RdWriteRow> {
    const EXTREMES: [i128; 4] = [u64::MAX as i128, -(u64::MAX as i128), i128::MAX, i128::MIN];
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..1usize << log_t)
        .map(|cycle| match cycle % 4 {
            0 => RdWriteRow {
                rd_inc: RdInc(EXTREMES[cycle / 4 % 4]),
                rd: Some(if cycle & 4 == 0 { 0 } else { 127 }),
            },
            2 => RdWriteRow {
                rd_inc: RdInc(0),
                rd: None,
            },
            _ => {
                let bits = next();
                let magnitude = i128::from(next());
                RdWriteRow {
                    rd_inc: RdInc(if bits & 1 == 0 { magnitude } else { -magnitude }),
                    rd: Some((bits >> 1) as u8 & 0x7F),
                }
            }
        })
        .collect()
}

/// The reference tier's kernel over `rows`: its naive prover on the dense
/// increment column and the reference fold of the rows' one-hot `(K × T)`
/// write-address grid.
fn synthetic_reference<F: JoltField>(
    rows: &[RdWriteRow],
    instance: &Instance<F>,
) -> NaiveSumcheckProver<F, RegistersValEvaluation<F>> {
    let log_t = rows.len().ilog2() as usize;
    let (r_address, r_cycle) = instance
        .points
        .registers_val
        .split_at(REGISTER_ADDRESS_BITS);
    let mut grid = vec![F::zero(); rows.len() << REGISTER_ADDRESS_BITS];
    for (cycle, row) in rows.iter().enumerate() {
        if let Some(register) = row.rd {
            grid[(usize::from(register) << log_t) | cycle] = F::one();
        }
    }
    let mut one_hot = FixedBackend::new();
    one_hot
        .insert(
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::RdWa),
            Shape::new(REGISTER_ADDRESS_BITS + log_t, PolynomialEncoding::Dense),
            grid,
        )
        .unwrap();
    let opening_tables = BTreeMap::from([
        (
            rd_wa_val_evaluation(),
            Polynomial::new(
                address_fold(&one_hot, rd_wa_val_evaluation(), log_t, r_address).unwrap(),
            ),
        ),
        (
            rd_inc_val_evaluation(),
            Polynomial::new(rows.iter().map(|row| row.rd_inc.to_field()).collect()),
        ),
    ]);
    let derived_tables = BTreeMap::from([(
        JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
        Polynomial::new(LtPolynomial::evaluations(r_cycle)),
    )]);
    NaiveSumcheckProver::new(
        &instance.inputs(),
        opening_tables,
        derived_tables,
        BindingOrder::LowToHigh,
    )
    .unwrap()
}

/// Drives `reference` and the streaming kernel over `rows`, its tables capped
/// at exactly their size after `depth` binds, through every round on
/// `round_challenges`: equal round polynomials, the dense tables first
/// present at round `c = clamp(depth, 1, log_t − 1)` with `T / 2^c` entries
/// each, equal output claims and the LT pin. Returns the input claim.
fn assert_capped_parity<F: JoltField>(
    reference: &mut Reference<F>,
    rows: BundleStore<RdWriteRow>,
    instance: &Instance<F>,
    round_challenges: &[F],
    depth: usize,
) -> F {
    let log_t = round_challenges.len();
    let cap = ((2 * size_of::<F>()) << log_t) >> depth;
    let mut streaming = StreamingValEvaluationKernel::new(rows, &instance.inputs(), cap).unwrap();
    let dense_len = |kernel: &StreamingValEvaluationKernel<F>| match &kernel.tables {
        ValueTables::Dense { inc, wa } => {
            assert_eq!(inc.len(), wa.len());
            Some(inc.len())
        }
        ValueTables::Streamed => None,
    };

    let input_claim = probe_input_claim(reference);
    let mut claim = input_claim;
    let mut filled = None;
    let mut bind = None;
    for (round, &challenge) in round_challenges.iter().enumerate() {
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        let streaming_poly = streaming.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            trimmed(&streaming_poly),
            trimmed(&reference_poly),
            "round {round} polynomial, log_t {log_t}, depth {depth}"
        );
        filled = filled.or_else(|| dense_len(&streaming).map(|len| (round, len)));
        claim = reference_poly.evaluate(challenge);
        bind = Some(challenge);
    }
    let last = bind.unwrap();
    reference.finish_rounds(last).unwrap();
    streaming.finish_rounds(last).unwrap();
    let crossover = depth.max(1).min(log_t - 1);
    assert_eq!(
        filled,
        Some((crossover, (1 << log_t) >> crossover)),
        "dense tables, log_t {log_t}, depth {depth}"
    );
    assert_eq!(
        streaming.output_claims(&instance.claims).unwrap(),
        reference.output_claims(&instance.claims).unwrap(),
        "output claims, log_t {log_t}, depth {depth}"
    );
    let output_points = instance
        .relation
        .derive_opening_points(round_challenges, &instance.points)
        .unwrap();
    streaming
        .validate_derived_tables(
            &instance.relation,
            &instance.points,
            &output_points,
            &instance.challenges,
        )
        .unwrap();
    input_claim
}

/// The crossover is exact at the cap's boundary, clamps to rounds
/// `1..=rounds − 1`, and does not wrap when the tables exceed `usize` bytes
/// (the widest cycle count `usize` holds).
#[test]
fn crossover_round_is_exact_at_the_cap_and_past_usize_bytes() {
    let cycle_bytes = 2 * size_of::<Fr>();
    assert_eq!(crossover_round::<Fr>(20, cycle_bytes << 10), 10);
    assert_eq!(crossover_round::<Fr>(20, (cycle_bytes << 10) - 1), 11);
    assert_eq!(crossover_round::<Fr>(20, usize::MAX), 1);
    assert_eq!(crossover_round::<Fr>(20, 0), 19);
    assert_eq!(crossover_round::<Fr>(1, 0), 0);
    let wide = usize::BITS as usize - 1;
    assert_eq!(crossover_round::<Fr>(wide, cycle_bytes << 20), wide - 20);
}

/// Full-range synthetic rows at every crossover round a cap selects, from a
/// cap the unbound tables meet (still deferred to round 1) to one no round
/// meets (the last round fills).
#[test]
fn synthetic_parity_at_every_crossover() {
    fn case<F: JoltField>(log_t: usize, seed: u64) {
        let rows = synthetic_rows(log_t, seed);
        let instance = Instance::<F>::new(
            log_t,
            challenge_sequence(REGISTER_ADDRESS_BITS + log_t, seed),
        );
        let round_challenges = challenge_sequence(log_t, seed ^ 0x5A5A);
        for depth in 0..=log_t {
            let claim = assert_capped_parity(
                &mut synthetic_reference(&rows, &instance),
                BundleStore::Retained(rows.clone()),
                &instance,
                &round_challenges,
                depth,
            );
            assert!(
                claim != F::zero() && claim != F::one(),
                "degenerate fixture"
            );
        }
    }
    for (log_t, seed) in [(1, 11), (2, 12), (3, 13), (4, 14), (6, 16)] {
        case::<Fr>(log_t, seed);
        #[cfg(feature = "akita")]
        case::<AkitaField>(log_t, seed);
    }
}

/// Zero and one as round challenges and as input-point coordinates: vanishing
/// contraction weights, a one-hot `eq(r_address, ·)` and LT tables with zero
/// entries, down to an identically zero sum.
#[test]
fn exceptional_challenges_and_points_match_reference() {
    fn case<F: JoltField>() {
        let log_t = 4;
        let rows = synthetic_rows(log_t, 41);
        let random: Vec<F> = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 43);
        let (zero, one) = (F::zero(), F::one());
        let mixed: Vec<F> = random
            .iter()
            .enumerate()
            .map(|(i, &value)| [zero, one, value][i % 3])
            .collect();
        let points = [
            vec![zero; REGISTER_ADDRESS_BITS + log_t],
            vec![one; REGISTER_ADDRESS_BITS + log_t],
            mixed,
        ];
        let challenge_sets = [
            vec![zero, one, zero, one],
            vec![one; log_t],
            vec![random[0], zero, random[1], one],
        ];
        for point in points {
            let instance = Instance::new(log_t, point);
            for round_challenges in &challenge_sets {
                for depth in 1..log_t {
                    let _ = assert_capped_parity(
                        &mut synthetic_reference(&rows, &instance),
                        BundleStore::Retained(rows.clone()),
                        &instance,
                        round_challenges,
                        depth,
                    );
                }
            }
        }
    }
    case::<Fr>();
    #[cfg(feature = "akita")]
    case::<AkitaField>();
}

/// The witness's resident rows (task boundaries and a padded proof suffix
/// inside the domain) against the reference kernel at every crossover round,
/// in both fields. At 2^13 cycles the last-round crossover's one pair spans
/// two visit callbacks.
#[test]
fn trace_rows_match_reference_at_every_crossover() {
    fn case<F: JoltField>(backend: &TraceBackend, log_t: usize, depth: usize) {
        let instance =
            Instance::<F>::new(log_t, challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 97));
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, instance.inputs())
            .unwrap();
        let _ = assert_capped_parity(
            reference.as_mut(),
            BundleStore::resident::<F>(backend, log_t, ROWS_LABEL).unwrap(),
            &instance,
            &challenge_sequence(log_t, 101),
            depth,
        );
    }
    for (log_t, cycles) in [(1, 1), (3, 5), (6, 61)] {
        structured_fixture(cycles).with_plane(log_t, |backend| {
            for depth in 1..=log_t {
                case::<Fr>(backend, log_t, depth);
                #[cfg(feature = "akita")]
                case::<AkitaField>(backend, log_t, depth);
            }
        });
    }
    structured_fixture((1 << 13) - 3).with_plane(13, |backend| {
        for depth in [1, 12] {
            case::<Fr>(backend, 13, depth);
        }
    });
}

/// An operand outside the register domain fails the first streamed round
/// with the kernel's row source named.
#[test]
fn row_failure_names_the_row_source() {
    let mut fixture = structured_fixture(5);
    fixture.unchecked_read(200);
    fixture.with_plane(3, |backend| {
        let instance = Instance::<Fr>::new(3, challenge_sequence(REGISTER_ADDRESS_BITS + 3, 7));
        let mut kernel = StreamingRegistersValEvaluation
            .prepare(&mut ProofSession::default(), backend, instance.inputs())
            .unwrap();
        assert!(matches!(
            kernel.prove_round(None, 0, Fr::from_u64(0)),
            Err(SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL })
        ));
    });
}

/// A resident witness and the same trace under a configured domain shorter
/// than it, which leaves no resident rows; chunk visits still serve its
/// `2^log_t` cycles.
fn with_resident_and_chunk_only<R>(
    log_t: usize,
    f: impl FnOnce(&TraceBackend, &TraceBackend) -> R,
) -> R {
    structured_fixture(7).with_plane(3, |resident| {
        let mut chunk_only = TraceBackend::new(
            resident.config.clone(),
            JoltVmWitnessInputs::new(
                &resident.program,
                &resident.preprocessing,
                resident.trace.clone(),
            ),
        );
        chunk_only.config.log_t = log_t;
        f(resident, &chunk_only)
    })
}

/// A witness without resident rows, or with another cycle domain, fails typed
/// instead of collecting rows.
#[test]
fn missing_or_mismatched_rows_are_typed() {
    let prepare = |witness: &TraceBackend, log_t: usize| {
        let instance =
            Instance::<Fr>::new(log_t, challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 5));
        StreamingRegistersValEvaluation
            .prepare(&mut ProofSession::default(), witness, instance.inputs())
            .map(|_| ())
    };
    with_resident_and_chunk_only(1, |resident, chunk_only| {
        assert!(matches!(
            prepare(chunk_only, 3),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(matches!(
            prepare(resident, 4),
            Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
        ));
        // A cycle count that does not fit `usize` is refused the same way.
        assert!(matches!(
            prepare(resident, usize::BITS as usize),
            Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
        ));
    });
}

/// `with_streaming` puts the family's kernels in the backend's slots.
/// Streamed and default proofs are byte-equal, so only the kernels' behavior
/// shows which ones are installed: the stage-5 slot and the stage-4 slot,
/// cycle-first here, refuse a witness without resident rows, which the
/// default kernels collect. Dory stands in for the commitment scheme; the
/// akita build's `optimized` takes another.
#[cfg(not(feature = "akita"))]
#[test]
fn with_streaming_installs_the_register_kernels() {
    use jolt_claims::protocols::jolt::ReadWriteDimensions;
    use jolt_dory::DoryScheme;
    use jolt_verifier::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
    };

    use crate::optimized::streaming::StreamingFamilies;
    use crate::JoltBackend;

    let log_t = 2;
    let streaming = || {
        JoltBackend::<Fr, DoryScheme>::optimized().with_streaming(StreamingFamilies {
            registers: true,
            ..StreamingFamilies::default()
        })
    };
    with_resident_and_chunk_only(log_t, |_, chunk_only| {
        let instance =
            Instance::<Fr>::new(log_t, challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 5));
        let val_evaluation = |backend: JoltBackend<Fr, DoryScheme>| {
            backend
                .registers_val_evaluation
                .prepare(&mut ProofSession::default(), chunk_only, instance.inputs())
                .map(|_| ())
        };
        val_evaluation(JoltBackend::optimized()).unwrap();
        assert!(matches!(
            val_evaluation(streaming()),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));

        let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
            log_t,
            REGISTER_ADDRESS_BITS,
            log_t,
            0,
        ));
        let claims = RegistersReadWriteInputClaims {
            rd_write_value: Fr::from_u64(0),
            rs1_value: Fr::from_u64(0),
            rs2_value: Fr::from_u64(0),
        };
        let point: Vec<Fr> = challenge_sequence(log_t, 9);
        let points = RegistersReadWriteInputClaims {
            rd_write_value: point.clone(),
            rs1_value: point.clone(),
            rs2_value: point,
        };
        let challenges = RegistersReadWriteChallenges {
            gamma: Fr::from_u64(3),
        };
        let read_write = |backend: JoltBackend<Fr, DoryScheme>| {
            backend
                .registers_read_write
                .prepare(
                    &mut ProofSession::default(),
                    chunk_only,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .map(|_| ())
        };
        read_write(JoltBackend::optimized()).unwrap();
        assert!(matches!(
            read_write(streaming()),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
    });
}
