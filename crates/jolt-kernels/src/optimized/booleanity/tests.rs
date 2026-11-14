use crate::optimized::parity::{probe_input_claim, run_lockstep, ExceptionalEq};
use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Fr, Ring, Zero};
use jolt_poly::EqPolynomial;
use jolt_riscv::JoltTraceRow as TraceRow;
use jolt_verifier::stages::relations::ConcreteSumcheckChallenges;
use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{BytecodePc, LookupIndex, RemappedRamAddress};
#[cfg(not(feature = "akita"))]
use jolt_witness::{JoltVmWitnessInputs, WitnessError};
use jolt_witness::{JoltWitnessOracle, TraceBackend};

use super::super::instruction_read_raf::{SharedInstructionRows, SharedInstructionRowsWeak};
use super::testing::{
    alu_row, load_row, no_op_row, store_row, test_challenge, with_booleanity_backend,
    with_trace_backend,
};
use super::*;
use crate::optimized::one_hot_histograms::HistogramRow;
use crate::optimized::support::BundleStore;
use crate::ReferenceBackend;

fn check_address_masses<F: JoltField>() {
    with_booleanity_backend(6, 4, |backend, dimensions| {
        let columns = BooleanityColumns::new::<Fr>(backend, dimensions).unwrap();
        let original = InstructionCycleRow::collect::<Fr>(backend, 64).unwrap();
        for prefix in [0, 1, 7, 8, 9, 33, 63] {
            let mut rows: Vec<_> = (0..original.len()).map(|i| original[i]).collect();
            rows[prefix..].fill(original[0]);
            for point in [
                (0..6).map(|i| F::from_u64(19 + 3 * i)).collect(),
                vec![F::zero(); 6],
                vec![F::one(); 6],
            ] {
                let weights = EqPolynomial::<F>::evals(&point, None);
                let expected: Vec<Vec<F>> = columns
                    .selectors
                    .iter()
                    .map(|selector| {
                        (0..16)
                            .map(|k| {
                                rows.iter()
                                    .zip(&weights)
                                    .filter(|(row, _)| selector.index(*row) == Some(k))
                                    .map(|(_, weight)| *weight)
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                let masses = cycle_pushforward(dimensions.layout, 16, |families| {
                    Ok(pushforwards(
                        &rows.clone().into(),
                        &point,
                        families,
                        16,
                        true,
                    )?)
                });
                assert_eq!(masses.unwrap(), expected);
            }
        }
    });
}

#[test]
fn address_masses_include_constant_suffix_at_exceptional_points() {
    check_address_masses::<Fr>();
}

#[cfg(feature = "akita")]
#[test]
fn address_masses_include_constant_suffix_at_exceptional_points_akita() {
    check_address_masses::<Prime128Offset6A23801F>();
}

#[expect(clippy::type_complexity)]
fn drive_lockstep<R>(
    mut reference: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    mut optimized: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    input_claim: Fr,
) -> (
    Box<dyn SumcheckKernel<Fr, Relation = R>>,
    Box<dyn SumcheckKernel<Fr, Relation = R>>,
    Vec<Fr>,
)
where
    R: ConcreteSumcheck<Fr>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr>,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let rounds = reference.num_rounds();
    assert_eq!(rounds, optimized.num_rounds());
    let mut claim = input_claim;
    let mut bind = None;
    let mut challenges = Vec::new();
    for round in 0..rounds {
        let expected = reference.prove_round(bind, round, claim).unwrap();
        let actual = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(expected, actual, "round {round} polynomial mismatch");
        let challenge = test_challenge(round);
        claim = expected.evaluate(challenge);
        challenges.push(challenge);
        bind = Some(challenge);
    }
    if let Some(last) = challenges.last() {
        reference.finish_rounds(*last).unwrap();
        optimized.finish_rounds(*last).unwrap();
    }
    (reference, optimized, challenges)
}

fn point<F: JoltField>(seed: u64, len: usize) -> Vec<F> {
    (0..len as u64)
        .map(|index| F::from_u64(seed + 37 * index + 5))
        .collect()
}

fn cycle_relation(
    dimensions: BooleanityDimensions,
    r_address: Vec<Fr>,
    reference_address: Vec<Fr>,
    reference_cycle: Vec<Fr>,
) -> Booleanity<Fr> {
    Booleanity::new(dimensions, r_address, reference_address, reference_cycle)
}

/// Brute-forces the cycle-phase input claim from the dense one-hot
/// grids: `Σ_j eq_rr · eq_cycle(j) · Σ_i γ^{2i} (x_i(j)² − x_i(j))` with
/// `x_i` the address-folded rows — independent of both kernels.
fn brute_force_cycle_claim(
    backend: &dyn JoltWitnessOracle<Fr>,
    dimensions: BooleanityDimensions,
    r_address: &[Fr],
    reference_address: &[Fr],
    reference_cycle: &[Fr],
    gamma: Fr,
) -> Fr {
    let eq_address = EqPolynomial::new(r_address.to_vec()).evaluations();
    let address_scalar = try_eq_mle(r_address, reference_address).unwrap();
    let eq_cycle = EqPolynomial::new(reference_cycle.to_vec()).evaluations();
    let log_t = dimensions.log_t;
    let gamma_sqr = gamma * gamma;
    let mut total = Fr::from_u64(0);
    let mut weight = Fr::from_u64(1);
    for opening in dimensions.layout.openings(JoltRelationId::Booleanity) {
        let grid: Vec<Fr> = backend.oracle_table(opening.polynomial_id()).unwrap();
        for (j, eq_cycle) in eq_cycle.iter().enumerate() {
            let x: Fr = eq_address
                .iter()
                .enumerate()
                .map(|(k, eq)| *eq * grid[(k << log_t) | j])
                .sum();
            total += address_scalar * *eq_cycle * weight * (x * x - x);
        }
        weight *= gamma_sqr;
    }
    total
}

fn address_parity(log_t: usize, log_k_chunk: u8) {
    with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
        let relation = BooleanityAddressPhase::<Fr>::new(
            dimensions,
            point(900, dimensions.log_k_chunk),
            point(300, log_t),
        );
        let claims = Default::default();
        let points = Default::default();
        let challenges = BooleanityAddressPhaseChallenges {
            reference_address: point(700, dimensions.log_k_chunk),
            gamma: Fr::from_u64(31),
        };
        let reference = ReferenceBackend
            .prepare(
                &mut ProofSession::default(),
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
        let mut session = ProofSession::default();
        let optimized = OptimizedBooleanityAddress
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
        let (mut reference, mut optimized, _) =
            drive_lockstep(reference, optimized, Fr::from_u64(0));
        assert_eq!(
            reference.output_claims(&claims).unwrap(),
            optimized.output_claims(&claims).unwrap(),
        );
        assert!(
            session.state::<SharedInstructionRows>().is_some()
                || session.state::<SharedInstructionRowsWeak>().is_some()
        );
    });
}

#[test]
fn address_kernel_matches_reference() {
    address_parity(2, 4);
}

#[test]
fn address_kernel_matches_reference_single_cycle_round() {
    address_parity(1, 4);
}

#[test]
fn address_kernel_matches_reference_wide_chunk() {
    address_parity(2, 8);
}

fn cycle_parity(log_t: usize, log_k_chunk: u8, carried_indices: bool) {
    cycle_parity_with_gamma(log_t, log_k_chunk, carried_indices, Fr::from_u64(31));
}

fn cycle_parity_with_gamma(log_t: usize, log_k_chunk: u8, carried_indices: bool, gamma: Fr) {
    cycle_parity_case(log_t, log_k_chunk, carried_indices, gamma, None);
}

fn cycle_parity_case(
    log_t: usize,
    log_k_chunk: u8,
    carried_indices: bool,
    gamma: Fr,
    exceptional: Option<ExceptionalEq>,
) {
    with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
        let r_address = point(110, dimensions.log_k_chunk);
        let reference_address = point(700, dimensions.log_k_chunk);
        let reference_cycle = exceptional.map_or_else(
            || point(400, log_t),
            |case| case.point(log_t, test_challenge(0)),
        );
        let relation = cycle_relation(
            dimensions,
            r_address.clone(),
            reference_address.clone(),
            reference_cycle.clone(),
        );
        let input_claim = brute_force_cycle_claim(
            backend,
            dimensions,
            &r_address,
            &reference_address,
            &reference_cycle,
            gamma,
        );
        let claims = BooleanityInputClaims {
            address_phase: input_claim,
        };
        let points = BooleanityInputClaims {
            address_phase: point(50, dimensions.log_k_chunk),
        };
        let challenges = BooleanityCyclePhaseChallenges { gamma };

        let reference = ReferenceBackend
            .prepare(
                &mut ProofSession::default(),
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
        let mut session = ProofSession::default();
        if carried_indices {
            let rows = InstructionCycleRow::collect::<Fr>(backend, 1usize << log_t).unwrap();
            session.park(SharedInstructionRows(Arc::new(rows)));
        }
        let optimized = OptimizedBooleanityCycle
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
        if carried_indices {
            assert!(
                session.state::<SharedInstructionRows>().is_some()
                    || session.state::<SharedInstructionRowsWeak>().is_some(),
                "cycle prepare must park a shared-rows carry back for later consumers"
            );
        }

        let (mut reference, mut optimized, challenges_drawn) =
            drive_lockstep(reference, optimized, input_claim);
        let reference_outputs = reference.output_claims(&claims).unwrap();
        let optimized_outputs = optimized.output_claims(&claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);
        if gamma.is_zero() {
            assert!(reference_outputs
                .opening_values()
                .iter()
                .skip(1)
                .any(|value| !value.is_zero()));
        }

        let output_points = relation
            .derive_opening_points(&challenges_drawn, &points)
            .unwrap();
        reference
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
    });
}

#[test]
fn cycle_kernel_matches_reference_at_exceptional_eq_points() {
    for bits in [4u8, 8] {
        for gamma in [Fr::from_u64(0), Fr::from_u64(31)] {
            for case in ExceptionalEq::ALL {
                cycle_parity_case(6, bits, true, gamma, Some(case));
            }
        }
    }
}

#[test]
fn cycle_kernel_matches_reference_with_zero_batching_gamma() {
    cycle_parity_with_gamma(5, 4, false, Fr::from_u64(0));
}

/// Round 0 reaches the categorical cache from `BooleanityChunks`: at
/// `log_t = 12` only the triangular table fits `T/16`.
#[test]
fn cycle_kernel_matches_reference_with_triangular_categories() {
    cycle_parity(12, 4, false);
}

/// At `log_t = 13` round 0 caches with the full table.
#[test]
fn cycle_kernel_matches_reference_with_full_categories() {
    cycle_parity(13, 4, false);
}

#[test]
fn cycle_kernel_matches_reference_single_round() {
    cycle_parity(1, 4, false);
}

/// `log_t = 5` drives the shared-table state machine through
/// materialization (four staged binds, dense at `T/16`) into a dense
/// round message. At `log_t = 4`, materialization happens only during
/// `finish_rounds`.
#[test]
fn cycle_kernel_matches_reference_through_dense_rounds() {
    cycle_parity(5, 4, false);
}

#[test]
fn cycle_kernel_matches_reference_materializing_at_finish() {
    cycle_parity(4, 4, false);
}

#[test]
fn cycle_kernel_matches_reference_wide_chunk() {
    cycle_parity(2, 8, false);
}

#[test]
fn cycle_kernel_matches_reference_with_carried_indices() {
    cycle_parity(2, 4, true);
}

/// The production 6a→6b sequencing: the address phase (both kernels in
/// lockstep) stages the intermediate claim; the cycle phase consumes it
/// with `r_address` = the reversed address challenges, reclaiming the
/// parked index columns from the shared session. The reference kernel's
/// round-zero check pins the intermediate claim to the cycle-phase sum,
/// so a cross-phase orientation drift fails loudly.
#[test]
fn two_phase_flow_matches_reference_end_to_end() {
    let log_t = 2;
    with_booleanity_backend(log_t, 4, |backend, dimensions| {
        let reference_address = point(700, dimensions.log_k_chunk);
        let reference_cycle = point(400, log_t);
        let gamma = Fr::from_u64(31);
        // The stage-6a relation derives its reference cycle from the
        // stage-5 instruction cycle by reversal; feed the reversed
        // vector so `reference_cycle()` equals the 6b relation's.
        let instruction_r_cycle: Vec<Fr> = reference_cycle.iter().rev().copied().collect();
        let address_relation = BooleanityAddressPhase::<Fr>::new(
            dimensions,
            point(900, dimensions.log_k_chunk),
            instruction_r_cycle,
        );
        let address_claims = Default::default();
        let address_points = Default::default();
        let address_challenges = BooleanityAddressPhaseChallenges {
            reference_address: reference_address.clone(),
            gamma,
        };

        let mut session = ProofSession::default();
        let reference = ReferenceBackend
            .prepare(
                &mut ProofSession::default(),
                backend,
                ProverInputs {
                    relation: &address_relation,
                    claims: &address_claims,
                    points: &address_points,
                    challenges: &address_challenges,
                },
            )
            .unwrap();
        let optimized = OptimizedBooleanityAddress
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &address_relation,
                    claims: &address_claims,
                    points: &address_points,
                    challenges: &address_challenges,
                },
            )
            .unwrap();
        let (mut reference, mut optimized, address_challenges_drawn) =
            drive_lockstep(reference, optimized, Fr::from_u64(0));
        let intermediate = reference.output_claims(&address_claims).unwrap();
        assert_eq!(
            intermediate,
            optimized.output_claims(&address_claims).unwrap()
        );

        let r_address: Vec<Fr> = address_challenges_drawn.iter().rev().copied().collect();
        let cycle_relation = cycle_relation(
            dimensions,
            r_address.clone(),
            reference_address.clone(),
            reference_cycle.clone(),
        );
        let input_claim = brute_force_cycle_claim(
            backend,
            dimensions,
            &r_address,
            &reference_address,
            &reference_cycle,
            gamma,
        );
        assert_eq!(intermediate.intermediate, input_claim);

        let cycle_claims = BooleanityInputClaims {
            address_phase: input_claim,
        };
        let cycle_points = BooleanityInputClaims {
            address_phase: r_address.clone(),
        };
        let cycle_challenges = BooleanityCyclePhaseChallenges { gamma };
        let reference = ReferenceBackend
            .prepare(
                &mut ProofSession::default(),
                backend,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_points,
                    challenges: &cycle_challenges,
                },
            )
            .unwrap();
        let optimized = OptimizedBooleanityCycle
            .prepare(
                &mut session,
                backend,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_points,
                    challenges: &cycle_challenges,
                },
            )
            .unwrap();
        assert!(
            session.state::<SharedInstructionRows>().is_some()
                || session.state::<SharedInstructionRowsWeak>().is_some(),
            "cycle prepare must park a shared-rows carry back"
        );
        let (mut reference, mut optimized, cycle_challenges_drawn) =
            drive_lockstep(reference, optimized, input_claim);
        assert_eq!(
            reference.output_claims(&cycle_claims).unwrap(),
            optimized.output_claims(&cycle_claims).unwrap()
        );
        let output_points = cycle_relation
            .derive_opening_points(&cycle_challenges_drawn, &cycle_points)
            .unwrap();
        optimized
            .validate_derived_tables(
                &cycle_relation,
                &cycle_points,
                &output_points,
                &cycle_challenges,
            )
            .unwrap();
    });
}

/// `len` rows of the fixture program: loads and stores over eleven RAM
/// addresses, ALU rows of varied operands, and no-ops.
fn mixed_rows(len: usize) -> Vec<TraceRow> {
    (0..len as u64)
        .map(|cycle| {
            let address = 0x8000_1000 + 8 * (cycle % 11);
            match cycle % 5 {
                0 => load_row(address),
                1 => alu_row(cycle.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                2 => store_row(address),
                3 => no_op_row(),
                _ => alu_row(cycle),
            }
        })
        .collect()
}

/// An address-phase relation over `dimensions` with its prover inputs.
struct AddressCase<F: JoltField> {
    relation: BooleanityAddressPhase<F>,
    claims: SumcheckInputClaims<F, BooleanityAddressPhase<F>>,
    points: SumcheckInputPoints<F, BooleanityAddressPhase<F>>,
    challenges: BooleanityAddressPhaseChallenges<F>,
}

impl<F: JoltField> AddressCase<F> {
    fn new(dimensions: BooleanityDimensions) -> Self {
        Self {
            relation: BooleanityAddressPhase::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                point(300, dimensions.log_t),
            ),
            claims: Default::default(),
            points: Default::default(),
            challenges: BooleanityAddressPhaseChallenges {
                reference_address: point(700, dimensions.log_k_chunk),
                gamma: F::from_u64(31),
            },
        }
    }

    fn inputs(&self) -> ProverInputs<'_, F, BooleanityAddressPhase<F>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// A cycle-phase relation over `dimensions` with its prover inputs. Kernels
/// take the input claim through their first round, not from `claims`.
struct CycleCase<F: JoltField> {
    relation: Booleanity<F>,
    claims: SumcheckInputClaims<F, Booleanity<F>>,
    points: SumcheckInputPoints<F, Booleanity<F>>,
    challenges: BooleanityCyclePhaseChallenges<F>,
}

impl<F: JoltField> CycleCase<F> {
    fn new(dimensions: BooleanityDimensions, gamma: F, reference_cycle: Vec<F>) -> Self {
        Self {
            relation: Booleanity::new(
                dimensions,
                point(110, dimensions.log_k_chunk),
                point(700, dimensions.log_k_chunk),
                reference_cycle,
            ),
            claims: BooleanityInputClaims {
                address_phase: F::zero(),
            },
            points: BooleanityInputClaims {
                address_phase: point(50, dimensions.log_k_chunk),
            },
            challenges: BooleanityCyclePhaseChallenges { gamma },
        }
    }

    fn inputs(&self) -> ProverInputs<'_, F, Booleanity<F>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }

    /// Every round of `kernel` against `oracle` from the input claim `claim`,
    /// then the output claims and `kernel`'s eq scalar.
    fn assert_parity(
        &self,
        claim: F,
        oracle: &mut dyn SumcheckKernel<F, Relation = Booleanity<F>>,
        kernel: &mut dyn SumcheckKernel<F, Relation = Booleanity<F>>,
    ) {
        let challenges = point(883, self.relation.rounds());
        run_lockstep(oracle, kernel, claim, &challenges);
        assert_eq!(
            kernel.output_claims(&self.claims).unwrap(),
            oracle.output_claims(&self.claims).unwrap(),
        );
        let output_points = self
            .relation
            .derive_opening_points(&challenges, &self.points)
            .unwrap();
        kernel
            .validate_derived_tables(
                &self.relation,
                &self.points,
                &output_points,
                &self.challenges,
            )
            .unwrap();
    }
}

fn streaming_address_case<F: JoltField>(backend: &TraceBackend, dimensions: BooleanityDimensions) {
    let case = AddressCase::<F>::new(dimensions);
    let mut reference = ReferenceBackend
        .prepare(&mut ProofSession::default(), backend, case.inputs())
        .unwrap();
    let mut streaming = StreamingBooleanityAddress
        .prepare(&mut ProofSession::default(), backend, case.inputs())
        .unwrap();
    // Both tables start from the same masses, so the input claim is zero.
    run_lockstep(
        reference.as_mut(),
        streaming.as_mut(),
        F::zero(),
        &point(883, dimensions.log_k_chunk),
    );
    assert_eq!(
        streaming.output_claims(&case.claims).unwrap(),
        reference.output_claims(&case.claims).unwrap(),
    );
}

/// Every round and the intermediate claim of the streaming address phase
/// over a witness's resident rows, against the reference, in both fields.
/// The masses weigh the rows before the padded suffix one by one and the
/// suffix by its first row, so the trace ends before its first cycle, after
/// it, inside the domain, one cycle short of it and at its end.
#[test]
fn streaming_address_kernel_matches_reference() {
    for (log_t, bits) in [(1, 4), (6, 4), (6, 8)] {
        for rows in [0, 1, (5 << log_t) / 8, (1 << log_t) - 1, 1 << log_t] {
            with_trace_backend(log_t, bits, mixed_rows(rows), |backend, dimensions| {
                streaming_address_case::<Fr>(backend, dimensions);
                #[cfg(feature = "akita")]
                streaming_address_case::<Prime128Offset6A23801F>(backend, dimensions);
            });
        }
    }
}

/// One trace-backed cycle-phase case against the reference: the streaming
/// kernel with each checked polynomial materialized within `bytes`, or the
/// streaming slot at the production budget when `None`.
fn streaming_cycle_case<F: JoltField>(
    backend: &TraceBackend,
    dimensions: BooleanityDimensions,
    gamma: F,
    reference_cycle: Vec<F>,
    bytes: Option<usize>,
) {
    let case = CycleCase::new(dimensions, gamma, reference_cycle);
    let mut reference = ReferenceBackend
        .prepare(&mut ProofSession::default(), backend, case.inputs())
        .unwrap();
    let mut streaming: Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>> = match bytes {
        Some(bytes) => Box::new(
            OptimizedBooleanityCycleKernel::streamed(
                backend,
                &case.inputs(),
                BundleStore::resident::<F>(backend, dimensions.log_t, "trace rows").unwrap(),
                bytes,
            )
            .unwrap(),
        ),
        None => StreamingBooleanityCycle
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap(),
    };
    let claim = probe_input_claim(reference.as_mut());
    case.assert_parity(claim, reference.as_mut(), streaming.as_mut());
}

/// Every round, the output claims and the eq scalar of the streaming cycle
/// phase over a witness's resident rows, against the reference, in both
/// fields: at every materializing bind a budget selects (the fourth through
/// the last, which leaves one entry per polynomial) and at the production
/// budget through the slot, with 8-bit chunks, and on domains shorter than
/// four binds. The traces fill their domain, end one cycle short of it, hold
/// no cycle, one or two, and end inside a pair and a split-eq group or on a
/// pair boundary.
#[test]
fn streaming_cycle_kernel_matches_reference_at_every_materialization() {
    for (log_t, bits, rows) in [
        (6, 4, 0),
        (6, 4, 1),
        (6, 4, 2),
        (6, 4, 37),
        (6, 4, 40),
        (6, 4, 63),
        (6, 4, 64),
        (6, 8, 40),
        (3, 4, 5),
        (3, 4, 8),
        (1, 4, 1),
        (1, 4, 2),
    ] {
        with_trace_backend(log_t, bits, mixed_rows(rows), |backend, dimensions| {
            for depth in (4..=log_t).map(Some).chain([None]) {
                streaming_cycle_case::<Fr>(
                    backend,
                    dimensions,
                    Fr::from_u64(31),
                    point(400, log_t),
                    depth.map(|depth| size_of::<Fr>() << (log_t - depth)),
                );
                #[cfg(feature = "akita")]
                streaming_cycle_case::<Prime128Offset6A23801F>(
                    backend,
                    dimensions,
                    Prime128Offset6A23801F::from_u64(31),
                    point(400, log_t),
                    depth.map(|depth| size_of::<Prime128Offset6A23801F>() << (log_t - depth)),
                );
            }
        });
    }
}

/// The streamed rounds under a zero batching challenge, which disables all
/// but the first checked polynomial, and at exceptional eq points, where
/// `q(1)` takes its own pass over the rows.
#[test]
fn streaming_cycle_kernel_matches_reference_at_exceptional_points() {
    let log_t = 6;
    // Two entries per polynomial: every round but the last is streamed.
    let bytes = Some(2 * size_of::<Fr>());
    with_trace_backend(log_t, 4, mixed_rows(40), |backend, dimensions| {
        streaming_cycle_case(backend, dimensions, Fr::zero(), point(400, log_t), bytes);
        for case in ExceptionalEq::ALL {
            for gamma in [Fr::zero(), Fr::from_u64(31)] {
                let reference_cycle = case.point(log_t, point::<Fr>(883, 1)[0]);
                streaming_cycle_case(backend, dimensions, gamma, reference_cycle, bytes);
            }
        }
    });
}

/// The streamed rounds over generated rows against the default kernel over
/// the same rows packed, at every materializing bind: lookup indices span
/// all 128 bits and RAM addresses both chunks, which the fixture program's
/// trace does not reach. The input claim comes from the summand's definition.
#[test]
fn streaming_cycle_kernel_matches_default_on_generated_rows() {
    let log_t = 6;
    with_booleanity_backend(log_t, 4, |backend, dimensions| {
        let rows: Vec<HistogramRow> = (0..1u64 << log_t)
            .map(|cycle| HistogramRow {
                lookup_index: LookupIndex(
                    u128::from(cycle + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
                ),
                bytecode_pc: BytecodePc((cycle % 4) as usize),
                remapped_ram_address: RemappedRamAddress(
                    (cycle % 3 != 0).then_some(cycle.wrapping_mul(0x9E37_79B9) % 64),
                ),
            })
            .collect();
        let packed: Vec<_> = rows
            .iter()
            .map(|row| {
                InstructionCycleRow::new(
                    row.lookup_index.0,
                    None,
                    false,
                    row.bytecode_pc.0,
                    row.remapped_ram_address.0,
                    #[cfg(feature = "akita")]
                    FusedInc(0),
                )
            })
            .collect();

        let case = CycleCase::new(dimensions, Fr::from_u64(31), point(400, log_t));
        let relation = &case.relation;
        let eq_address = EqPolynomial::new(relation.r_address().to_vec()).evaluations();
        let eq_cycle = EqPolynomial::new(relation.reference_cycle().to_vec()).evaluations();
        let mut weight = try_eq_mle(relation.r_address(), relation.reference_address()).unwrap();
        let mut claim = Fr::zero();
        for selector in &BooleanityColumns::new::<Fr>(backend, dimensions)
            .unwrap()
            .selectors
        {
            for (row, eq) in rows.iter().zip(&eq_cycle) {
                let x = selector.index(row).map_or(Fr::zero(), |k| eq_address[k]);
                claim += weight * *eq * (x * x - x);
            }
            weight *= case.challenges.gamma.square();
        }

        for depth in 4..=log_t {
            let mut session = ProofSession::default();
            session.park(SharedInstructionRows(Arc::new(packed.clone().into())));
            let mut default = OptimizedBooleanityCycle
                .prepare(&mut session, backend, case.inputs())
                .unwrap();
            let mut streaming = OptimizedBooleanityCycleKernel::streamed(
                backend,
                &case.inputs(),
                BundleStore::Retained(rows.clone()),
                size_of::<Fr>() << (log_t - depth),
            )
            .unwrap();
            case.assert_parity(claim, default.as_mut(), &mut streaming);
        }
    });
}

/// A one-cycle domain has no rounds: preparation gathers the rows dense, so
/// the output claims and the eq check are available at once.
#[test]
fn streaming_cycle_kernel_opens_a_zero_round_domain_like_the_reference() {
    with_trace_backend(0, 4, mixed_rows(1), |backend, dimensions| {
        let case = CycleCase::new(dimensions, Fr::from_u64(31), Vec::new());
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap();
        let mut streaming = StreamingBooleanityCycle
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap();
        assert_eq!(streaming.num_rounds(), 0);
        assert_eq!(
            streaming.output_claims(&case.claims).unwrap(),
            reference.output_claims(&case.claims).unwrap(),
        );
        let output_points = case
            .relation
            .derive_opening_points(&[], &case.points)
            .unwrap();
        streaming
            .validate_derived_tables(
                &case.relation,
                &case.points,
                &output_points,
                &case.challenges,
            )
            .unwrap();
    });
}

/// `with_streaming` puts the streaming kernels in both booleanity slots.
/// Streamed and default proofs are byte-equal, so only a kernel's behavior
/// shows which one is installed: each refuses a witness without resident
/// rows, which the default kernels collect. Dory stands in for the
/// commitment scheme; the akita build's `optimized` takes another.
#[cfg(not(feature = "akita"))]
#[test]
fn with_streaming_installs_both_booleanity_kernels() {
    use jolt_dory::DoryScheme;

    use crate::optimized::streaming::StreamingFamilies;
    use crate::JoltBackend;

    let log_t = 1;
    with_trace_backend(3, 4, mixed_rows(7), |resident, dimensions| {
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
        chunk_only.config.log_t = log_t;
        let dimensions =
            BooleanityDimensions::new(dimensions.layout, log_t, dimensions.log_k_chunk);
        let address = AddressCase::<Fr>::new(dimensions);
        let cycle = CycleCase::<Fr>::new(dimensions, Fr::from_u64(31), point(400, log_t));
        let prepare = |backend: JoltBackend<Fr, DoryScheme>| {
            (
                backend
                    .booleanity_address
                    .prepare(&mut ProofSession::default(), &chunk_only, address.inputs())
                    .map(|_| ()),
                backend
                    .booleanity_cycle
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
