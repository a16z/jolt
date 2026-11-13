use crate::optimized::parity::ExceptionalEq;
use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Fr, Ring, Zero};
use jolt_poly::EqPolynomial;
use jolt_verifier::stages::relations::ConcreteSumcheckChallenges;
use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
use jolt_witness::JoltWitnessOracle;

use super::super::instruction_read_raf::{SharedInstructionRows, SharedInstructionRowsWeak};
use super::testing::{test_challenge, with_booleanity_backend};
use super::*;
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
                                    .filter(|(row, _)| selector.index(row) == Some(k))
                                    .map(|(_, weight)| *weight)
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                assert_eq!(
                    cycle_pushforward(&rows.clone().into(), dimensions.layout, 16, &point).unwrap(),
                    expected,
                );
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

fn point(seed: u64, len: usize) -> Vec<Fr> {
    (0..len as u64)
        .map(|index| Fr::from_u64(seed + 37 * index + 5))
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
