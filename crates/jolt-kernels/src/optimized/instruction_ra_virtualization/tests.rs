use crate::optimized::parity::{probe_input_claim, run_lockstep, ExceptionalEq};
use std::sync::Arc;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

#[cfg(feature = "akita")]
use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::instruction::{
    committed_instruction_ra, InstructionRaVirtualizationDimensions,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionRaVirtualizationChallenges, InstructionRaVirtualizationInputClaims,
};
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::{Fr, JoltField, Ring};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
use jolt_witness::{FixedBackend, PolynomialEncoding, Shape, TraceBackend};
#[cfg(not(feature = "akita"))]
use jolt_witness::{JoltVmWitnessInputs, WitnessError};

use crate::optimized::registers_read_write::test_support::structured_fixture;
use crate::optimized::streaming::MATERIALIZE_BYTES;
use crate::optimized::support::BundleStore;
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::reference::views::{address_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};

use super::super::instruction_read_raf::{
    InstructionCycleRow, LookupIndexRow, SharedInstructionRows, SharedInstructionRowsWeak,
};
use super::super::support::MAX_GRID_FACTORS;
use super::super::testing::{with_ram_fixture, FixtureShape};
use super::{
    OptimizedInstructionRaVirtualization, OptimizedInstructionRaVirtualizationKernel,
    StreamingInstructionRaVirtualization,
};

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
    assert_parity_with(
        log_t,
        num_virtual,
        per_virtual,
        chunk_bits,
        seed,
        gamma,
        exceptional,
        prefix,
        |inputs, rows| {
            if with_session {
                // The witness plane comes from an unrelated trace fixture: a
                // missed take of the parked rows would stream that trace's
                // lookup indices instead and fail the parity loop loudly.
                let shape = FixtureShape { log_t, ram_k: 16 };
                with_ram_fixture(shape, Vec::new(), |witness| {
                    let mut session = ProofSession::default();
                    session.park(SharedInstructionRows(Arc::new(pack(rows).into())));
                    let kernel = OptimizedInstructionRaVirtualization
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: inputs.relation,
                                claims: inputs.claims,
                                points: inputs.points,
                                challenges: inputs.challenges,
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
                        inputs.relation,
                        Arc::new(pack(rows).into()),
                        gamma,
                    )
                    .unwrap(),
                )
            }
        },
    );
}

/// [`assert_parity_case`] for the streaming family over the rows, each
/// selector materialized within `bytes`.
#[expect(
    clippy::too_many_arguments,
    reason = "test shape plus exceptional equality case"
)]
fn assert_streaming_parity(
    log_t: usize,
    num_virtual: usize,
    per_virtual: usize,
    chunk_bits: usize,
    seed: u64,
    gamma: Fr,
    exceptional: Option<ExceptionalEq>,
    bytes: usize,
) {
    assert_parity_with(
        log_t,
        num_virtual,
        per_virtual,
        chunk_bits,
        seed,
        gamma,
        exceptional,
        None,
        |inputs, rows| {
            let rows = rows
                .iter()
                .map(|row| LookupIndexRow {
                    lookup_index: row.lookup_index,
                })
                .collect();
            Box::new(
                OptimizedInstructionRaVirtualizationKernel::streamed(
                    inputs.relation,
                    gamma,
                    BundleStore::Retained(rows),
                    bytes,
                )
                .unwrap(),
            )
        },
    );
}

/// [`assert_parity_case`] over the kernel `build` returns for the case's
/// inputs and rows.
#[expect(
    clippy::too_many_arguments,
    reason = "test shape plus exceptional equality case"
)]
fn assert_parity_with(
    log_t: usize,
    num_virtual: usize,
    per_virtual: usize,
    chunk_bits: usize,
    seed: u64,
    gamma: Fr,
    exceptional: Option<ExceptionalEq>,
    prefix: Option<usize>,
    build: impl FnOnce(
        &ProverInputs<'_, Fr, InstructionRaVirtualization<Fr>>,
        &[InstructionReadRafWitness],
    ) -> Box<dyn SumcheckKernel<Fr, Relation = InstructionRaVirtualization<Fr>>>,
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

    let mut optimized = build(&inputs, &rows);

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
        let relation = InstructionRaVirtualization::new(
            InstructionRaVirtualizationDimensions::new(
                log_t,
                NonZeroUsize::MIN,
                NonZeroUsize::new(per_virtual).unwrap(),
            )
            .unwrap(),
            (0..per_virtual as u64).map(fr).collect(),
            vec![fr(3), fr(5)],
            1,
        );
        let kernel = OptimizedInstructionRaVirtualizationKernel::new(
            &relation,
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

/// The streaming family against the reference at every materializing bind a
/// budget selects at 2^7 cycles (the fourth through the last, which leaves
/// one entry per selector), at the production budget, with eight factors
/// and with one, on domains shorter than four binds, under a zero batching
/// challenge, and at exceptional eq points, where `q(0)` takes its own pass.
#[test]
fn streaming_parity_at_every_materialization() {
    let entry = size_of::<Fr>();
    let gamma = fr(0xFEED_5EED);
    for depth in 4..=7 {
        assert_streaming_parity(7, 2, 2, 4, 61, gamma, None, entry << (7 - depth));
    }
    assert_streaming_parity(4, 8, 4, 4, 42, gamma, None, MATERIALIZE_BYTES);
    assert_streaming_parity(6, 4, 8, 4, 43, gamma, None, entry);
    assert_streaming_parity(3, 3, 1, 2, 1337, gamma, None, entry);
    assert_streaming_parity(1, 2, 2, 2, 9, gamma, None, entry);
    assert_streaming_parity(6, 2, 2, 4, 7, fr(0), None, entry);
    assert_streaming_parity(3, 3, 1, 2, 1337, fr(0), None, entry);
    for factors in [1, 2, 4] {
        for case in ExceptionalEq::ALL {
            assert_streaming_parity(6, 2, factors, 2, 277, fr(31), Some(case), 2 * entry);
        }
    }
}

/// The relation and prover inputs over the trace fixture's geometry: 4-bit
/// committed chunks of a 128-bit lookup index, four per virtual polynomial.
struct TraceCase<F: JoltField> {
    relation: InstructionRaVirtualization<F>,
    claims: InstructionRaVirtualizationInputClaims<F>,
    points: InstructionRaVirtualizationInputClaims<Vec<F>>,
    challenges: InstructionRaVirtualizationChallenges<F>,
}

impl<F: JoltField> TraceCase<F> {
    fn new(log_t: usize) -> Self {
        let dimensions = InstructionRaVirtualizationDimensions::new(
            log_t,
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        )
        .unwrap();
        Self {
            relation: InstructionRaVirtualization::new(
                dimensions,
                (0..128).map(|i| F::from_u64(300 + 13 * i)).collect(),
                (0..log_t as u64)
                    .map(|i| F::from_u64(7000 + 29 * i))
                    .collect(),
                4,
            ),
            claims: InstructionRaVirtualizationInputClaims {
                instruction_ra: vec![F::zero(); 8],
            },
            points: InstructionRaVirtualizationInputClaims {
                instruction_ra: vec![Vec::new(); 8],
            },
            challenges: InstructionRaVirtualizationChallenges {
                gamma: F::from_u64(0xFEED_5EED),
            },
        }
    }

    fn inputs(&self) -> ProverInputs<'_, F, InstructionRaVirtualization<F>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// One [`streaming_trace_rows_match_reference`] case: each selector
/// materialized within `bytes`, or within the production budget through the
/// slot when `None`.
fn trace_case<F: JoltField>(backend: &TraceBackend, log_t: usize, bytes: Option<usize>) {
    let case = TraceCase::<F>::new(log_t);
    let mut reference = ReferenceBackend
        .prepare(&mut ProofSession::default(), backend, case.inputs())
        .unwrap();
    let mut streaming: Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>> =
        match bytes {
            Some(bytes) => Box::new(
                OptimizedInstructionRaVirtualizationKernel::streamed(
                    &case.relation,
                    case.challenges.gamma,
                    BundleStore::resident::<F>(backend, log_t, "trace rows").unwrap(),
                    bytes,
                )
                .unwrap(),
            ),
            None => StreamingInstructionRaVirtualization
                .prepare(&mut ProofSession::default(), backend, case.inputs())
                .unwrap(),
        };
    let input_claim = probe_input_claim(reference.as_mut());
    let round_challenges: Vec<F> = (0..log_t as u64)
        .map(|i| F::from_u64(883 + 29 * i))
        .collect();
    run_lockstep(
        reference.as_mut(),
        streaming.as_mut(),
        input_claim,
        &round_challenges,
    );
    assert_eq!(
        streaming
            .output_claims(&case.claims)
            .unwrap()
            .committed_instruction_ra,
        reference
            .output_claims(&case.claims)
            .unwrap()
            .committed_instruction_ra,
        "output claims, log_t {log_t}, bytes {bytes:?}"
    );
}

/// Every round and the output claims over a witness's resident rows, whose
/// domain ends in a padded suffix of three eighths of its cycles, at every
/// materializing bind and at the production budget through the slot,
/// against the reference, in both fields.
#[test]
fn streaming_trace_rows_match_reference() {
    for log_t in [1, 3, 6] {
        structured_fixture(((5 << log_t) / 8).max(1)).with_plane(log_t, |backend| {
            for depth in (4..=log_t).map(Some).chain([None]) {
                trace_case::<Fr>(
                    backend,
                    log_t,
                    depth.map(|depth| size_of::<Fr>() << (log_t - depth)),
                );
                #[cfg(feature = "akita")]
                trace_case::<AkitaField>(
                    backend,
                    log_t,
                    depth.map(|depth| size_of::<AkitaField>() << (log_t - depth)),
                );
            }
        });
    }
}

/// A one-cycle domain has no rounds: preparation gathers the selectors
/// dense, so the output claims and the eq check are available at once.
#[test]
fn streaming_zero_round_domain_opens_like_the_reference() {
    let case = TraceCase::<Fr>::new(0);
    structured_fixture(1).with_plane(0, |backend| {
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap();
        let mut streaming = StreamingInstructionRaVirtualization
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap();
        assert_eq!(streaming.num_rounds(), 0);
        assert_eq!(
            streaming
                .output_claims(&case.claims)
                .unwrap()
                .committed_instruction_ra,
            reference
                .output_claims(&case.claims)
                .unwrap()
                .committed_instruction_ra,
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

/// `with_streaming` puts this kernel in the backend's slot. Streamed and
/// default proofs are byte-equal, so only the kernel's behavior shows which
/// one is installed: it refuses a witness without resident rows, which the
/// default kernel collects. Dory stands in for the commitment scheme; the
/// akita build's `optimized` takes another.
#[cfg(not(feature = "akita"))]
#[test]
fn with_streaming_installs_the_instruction_ra_virtualization_kernel() {
    use jolt_dory::DoryScheme;

    use crate::optimized::streaming::StreamingFamilies;
    use crate::JoltBackend;

    let log_t = 1;
    let case = TraceCase::<Fr>::new(log_t);
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
        chunk_only.config.log_t = log_t;
        let prepare = |backend: JoltBackend<Fr, DoryScheme>| {
            backend
                .instruction_ra_virtualization
                .prepare(&mut ProofSession::default(), &chunk_only, case.inputs())
                .map(|_| ())
        };
        prepare(JoltBackend::optimized()).unwrap();
        assert!(matches!(
            prepare(JoltBackend::optimized().with_streaming(StreamingFamilies {
                instruction_rows: true,
                ..StreamingFamilies::default()
            })),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
    });
}
