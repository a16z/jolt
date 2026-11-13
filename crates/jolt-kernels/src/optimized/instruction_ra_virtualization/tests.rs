use crate::optimized::parity::ExceptionalEq;
use std::sync::Arc;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::instruction::{
    committed_instruction_ra, InstructionRaVirtualizationDimensions,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionRaVirtualizationChallenges, InstructionRaVirtualizationInputClaims,
};
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::reference::views::{address_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
};

use super::super::instruction_read_raf::{
    InstructionCycleRow, SharedInstructionRows, SharedInstructionRowsWeak,
};
use super::super::support::MAX_GRID_FACTORS;
use super::super::testing::{with_ram_fixture, FixtureShape};
use super::{OptimizedInstructionRaVirtualization, OptimizedInstructionRaVirtualizationKernel};

#[test]
fn constant_suffix_matches_dense_rounds() {
    for prefix in [0, 1, 15, 16, 17, 31, 32, 33, 63] {
        for factors in [1, 2, 4, 8] {
            for exceptional in [None].into_iter().chain(ExceptionalEq::ALL.map(Some)) {
                assert_parity_case(
                    6,
                    2,
                    factors,
                    2,
                    277,
                    true,
                    fr(31),
                    exceptional,
                    Some(prefix),
                );
            }
        }
    }
}

fn pack(rows: &[InstructionReadRafWitness]) -> Vec<InstructionCycleRow> {
    rows.iter()
        .map(|row| {
            InstructionCycleRow::new(
                row.lookup_index.0,
                row.table_index.0,
                row.raf_flag.0,
                0,
                None,
                #[cfg(feature = "akita")]
                FusedInc::default(),
            )
        })
        .collect()
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn challenge(round: usize) -> Fr {
    fr(0xD1B5_4A32_D192_ED03 ^ (round as u64).wrapping_mul(0x94D0_49BB_1331_11EB) ^ 3)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
    let mut state = seed;
    (0..1usize << log_t)
        .map(|j| {
            let lookup_index = match j {
                0 => 0u128,
                1 => u128::MAX,
                _ if (8..16).contains(&(j % 32)) => u128::from(splitmix(&mut state) & 3),
                _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
            };
            InstructionReadRafWitness {
                lookup_index: LookupIndex(lookup_index),
                table_index: TableIndex(None),
                raf_flag: InstructionRafFlag(false),
            }
        })
        .collect()
}

/// Builds the committed one-hot `(K × T)` grid for chunk `i` exactly as
/// the trace backend serves it: address-major, hot at that chunk of the
/// cycle's lookup index, every cycle hot.
fn one_hot_grid(
    rows: &[InstructionReadRafWitness],
    chunk_index: usize,
    num_committed: usize,
    chunk_bits: usize,
) -> Vec<Fr> {
    let k = 1usize << chunk_bits;
    let t = rows.len();
    let shift = (num_committed - 1 - chunk_index) * chunk_bits;
    let mask = (1u128 << chunk_bits) - 1;
    let mut grid = vec![fr(0); k * t];
    for (j, row) in rows.iter().enumerate() {
        let hot = ((row.lookup_index.0 >> shift) & mask) as usize;
        grid[(hot * t) | j] = fr(1);
    }
    grid
}

/// Reference (naive prover over address-folded oracle grids, exactly as
/// the reference `prepare` assembles it) vs the optimized kernel, same
/// challenges: byte-equal round polynomials and output claims, and the
/// optimized eq-scalar passes the derived-table cross-check.
///
/// `with_session` builds the optimized kernel through the `prepare` slot
/// with pre-parked stage-5 rows instead of direct construction,
/// exercising the session take/park-back carry.
fn assert_parity(
    log_t: usize,
    num_virtual: usize,
    per_virtual: usize,
    chunk_bits: usize,
    seed: u64,
    with_session: bool,
) {
    assert_parity_with_gamma(
        log_t,
        num_virtual,
        per_virtual,
        chunk_bits,
        seed,
        with_session,
        fr(0xFEED_5EED),
    );
}

fn assert_parity_with_gamma(
    log_t: usize,
    num_virtual: usize,
    per_virtual: usize,
    chunk_bits: usize,
    seed: u64,
    with_session: bool,
    gamma: Fr,
) {
    assert_parity_case(
        log_t,
        num_virtual,
        per_virtual,
        chunk_bits,
        seed,
        with_session,
        gamma,
        None,
        None,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "test shape plus exceptional equality case"
)]
fn assert_parity_case(
    log_t: usize,
    num_virtual: usize,
    per_virtual: usize,
    chunk_bits: usize,
    seed: u64,
    with_session: bool,
    gamma: Fr,
    exceptional: Option<ExceptionalEq>,
    prefix: Option<usize>,
) {
    let num_committed = num_virtual * per_virtual;
    let dimensions = InstructionRaVirtualizationDimensions::new(
        log_t,
        NonZeroUsize::new(num_virtual).unwrap(),
        NonZeroUsize::new(per_virtual).unwrap(),
    )
    .unwrap();
    let mut rows = fixture_rows(log_t, seed);
    if let Some(prefix) = prefix {
        for row in &mut rows[prefix..] {
            row.lookup_index = LookupIndex(0);
        }
    }
    let instruction_address: Vec<Fr> = (0..num_committed * chunk_bits)
        .map(|i| fr(300 + 13 * i as u64))
        .collect();
    let r_cycle: Vec<Fr> = exceptional.map_or_else(
        || (0..log_t).map(|i| fr(7000 + 29 * i as u64)).collect(),
        |case| case.point(log_t, challenge(0)),
    );
    let relation = InstructionRaVirtualization::<Fr>::new(
        dimensions,
        instruction_address.clone(),
        r_cycle.clone(),
        chunk_bits,
    );

    let mut backend = FixedBackend::new();
    for index in 0..num_committed {
        let grid = one_hot_grid(&rows, index, num_committed, chunk_bits);
        backend
            .insert(
                committed_instruction_ra(index).polynomial_id(),
                Shape::new(chunk_bits + log_t, PolynomialEncoding::Dense),
                grid,
            )
            .unwrap();
    }
    let chunks = committed_address_chunks(&instruction_address, chunk_bits);
    let mut opening_tables = BTreeMap::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let folded =
            address_fold::<Fr>(&backend, committed_instruction_ra(index), log_t, chunk).unwrap();
        let _ = opening_tables.insert(committed_instruction_ra(index), Polynomial::new(folded));
    }
    let derived_tables = BTreeMap::from([(
        JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
        Polynomial::new(eq_table(&r_cycle)),
    )]);

    let input_claims = InstructionRaVirtualizationInputClaims {
        instruction_ra: vec![fr(0); num_virtual],
    };
    let input_points = InstructionRaVirtualizationInputClaims {
        instruction_ra: vec![Vec::new(); num_virtual],
    };
    let challenges = InstructionRaVirtualizationChallenges { gamma };
    let inputs = ProverInputs {
        relation: &relation,
        claims: &input_claims,
        points: &input_points,
        challenges: &challenges,
    };
    let mut reference = NaiveSumcheckProver::new(
        &inputs,
        opening_tables,
        derived_tables,
        BindingOrder::LowToHigh,
    )
    .unwrap();

    let mut optimized: Box<dyn SumcheckKernel<Fr, Relation = InstructionRaVirtualization<Fr>>> =
        if with_session {
            // The witness plane comes from an unrelated trace fixture: a
            // missed take of the parked rows would stream that trace's
            // lookup indices instead and fail the parity loop loudly.
            let shape = FixtureShape { log_t, ram_k: 16 };
            with_ram_fixture(shape, Vec::new(), |witness| {
                let mut session = ProofSession::default();
                session.park(SharedInstructionRows(Arc::new(pack(&rows).into())));
                let kernel = OptimizedInstructionRaVirtualization
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &input_claims,
                            points: &input_points,
                            challenges: &challenges,
                        },
                    )
                    .unwrap();
                assert!(
                    session.state::<SharedInstructionRows>().is_some()
                        || session.state::<SharedInstructionRowsWeak>().is_some(),
                    "prepare must park a shared-rows carry back for later consumers"
                );
                kernel
            })
        } else {
            Box::new(
                OptimizedInstructionRaVirtualizationKernel::new(
                    log_t,
                    num_virtual,
                    per_virtual,
                    &instruction_address,
                    &r_cycle,
                    chunk_bits,
                    Arc::new(pack(&rows).into()),
                    gamma,
                )
                .unwrap(),
            )
        };

    let eq_cycle = eq_table(&r_cycle);
    let mut claim = fr(0);
    for j in 0..rows.len() {
        let mut sum = fr(0);
        let mut gamma_power = fr(1);
        for v in 0..num_virtual {
            let mut product = fr(1);
            for i in 0..per_virtual {
                let index = v * per_virtual + i;
                let shift = (num_committed - 1 - index) * chunk_bits;
                let hot =
                    ((rows[j].lookup_index.0 >> shift) & ((1u128 << chunk_bits) - 1)) as usize;
                product *= eq_table(&chunks[index])[hot];
            }
            sum += gamma_power * product;
            gamma_power *= gamma;
        }
        claim += eq_cycle[j] * sum;
    }

    let rounds = reference.num_rounds();
    assert_eq!(rounds, optimized.num_rounds());
    assert_eq!(rounds, log_t);
    for round in 0..rounds {
        let bind = round.checked_sub(1).map(challenge);
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            reference_poly.coefficients(),
            optimized_poly.coefficients(),
            "round {round} polynomial mismatch (log_t={log_t}, V={num_virtual}, N={per_virtual})"
        );
        claim = reference_poly.evaluate(challenge(round));
    }
    reference.finish_rounds(challenge(rounds - 1)).unwrap();
    optimized.finish_rounds(challenge(rounds - 1)).unwrap();

    let reference_outputs = reference.output_claims(&input_claims).unwrap();
    let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
    assert_eq!(
        reference_outputs.committed_instruction_ra,
        optimized_outputs.committed_instruction_ra
    );
    if gamma == fr(0) {
        assert!(reference_outputs
            .committed_instruction_ra
            .iter()
            .skip(per_virtual)
            .any(|value| *value != fr(0)));
    }

    let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
    let output_points = relation
        .derive_opening_points(&sumcheck_point, &input_points)
        .unwrap();
    optimized
        .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
        .unwrap();
    reference
        .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
        .unwrap();
}

/// Production shape: 8 virtuals × 4 committed each, 4-bit chunks (the
/// 128-bit instruction address).
#[test]
fn parity_categorical_quartics() {
    assert_parity(10, 8, 4, 4, 933, false);
    for gamma in [fr(0), fr(1), fr(31)] {
        for prefix in [0, 513, 1023] {
            assert_parity_case(10, 2, 4, 2, 377, false, gamma, None, Some(prefix));
        }
    }
    for case in ExceptionalEq::ALL {
        assert_parity_case(10, 2, 4, 2, 377, false, fr(31), Some(case), Some(513));
    }
}

#[test]
fn paired_gathers_preserve_rounds_and_exceptional_eq() {
    for exceptional in [None].into_iter().chain(ExceptionalEq::ALL.map(Some)) {
        assert_parity_case(
            15,
            1,
            4,
            4,
            388,
            false,
            fr(31),
            exceptional,
            Some((1 << 15) - 5),
        );
    }
}

#[test]
fn parity_production_geometry() {
    assert_parity(4, 8, 4, 4, 42, false);
}

#[test]
fn parity_eight_factors_past_lazy_materialization() {
    assert_parity(6, 4, 8, 4, 43, false);
}

/// The product grid's factor bound is a construction error in every
/// build profile, not a debug assertion in the round loop.
#[test]
fn rejects_more_factors_than_the_product_grid_supports() {
    for (per_virtual, supported) in [(MAX_GRID_FACTORS, true), (MAX_GRID_FACTORS + 1, false)] {
        let log_t = 2;
        let instruction_address: Vec<Fr> = (0..per_virtual as u64).map(fr).collect();
        let kernel = OptimizedInstructionRaVirtualizationKernel::new(
            log_t,
            1,
            per_virtual,
            &instruction_address,
            &[fr(3), fr(5)],
            1,
            Arc::new(pack(&fixture_rows(log_t, 47)).into()),
            fr(7),
        );
        if supported {
            assert!(
                kernel.is_ok(),
                "{per_virtual} committed chunks per virtual polynomial"
            );
        } else {
            assert!(matches!(kernel, Err(KernelError::Unsupported { .. })));
        }
    }
}

#[test]
fn parity_small_odd_geometry() {
    assert_parity(3, 3, 2, 2, 1337, false);
}

#[test]
fn parity_past_lazy_materialization() {
    // log_t = 6: three lazy binds, dense materialization at the fourth
    // (`T/16` = 4 entries), then two plain multilinear binds.
    assert_parity(6, 2, 2, 4, 7, false);
}

#[test]
fn parity_zero_batching_challenge_preserves_all_openings() {
    assert_parity_with_gamma(6, 2, 2, 4, 7, false, fr(0));
    assert_parity_with_gamma(3, 3, 1, 2, 1337, false, fr(0));
}

/// Through the `prepare` slot with pre-parked stage-5 rows: the session
/// take/park-back carry serves this kernel the same rows and parks them
/// back for the stage-6a/6b booleanity consumers.
#[test]
fn parity_with_carried_session_rows() {
    assert_parity(4, 8, 4, 4, 42, true);
}
#[test]
fn parity_exceptional_eq_in_lazy_and_dense_virtualization() {
    for factors in [1usize, 2, 4] {
        for case in ExceptionalEq::ALL {
            assert_parity_case(6, 2, factors, 2, 277, true, fr(31), Some(case), None);
        }
    }
}
