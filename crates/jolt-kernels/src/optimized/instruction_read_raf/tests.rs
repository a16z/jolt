use crate::optimized::parity::{probe_input_claim, run_lockstep, ExceptionalEq};
use std::num::NonZeroUsize;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafInputClaims,
};
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Fr, JoltField, Ring};
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::testing::with_sample_backend;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
#[cfg(not(feature = "akita"))]
use jolt_witness::{JoltVmWitnessInputs, WitnessError};
use jolt_witness::{RowSource, TraceBackend};
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

use crate::optimized::registers_read_write::test_support::structured_fixture;
use crate::optimized::streaming::VISIT_ROWS;
use crate::optimized::support::BundleStore;
use crate::reference::instruction_read_raf::{InstructionReadRafKernel, InstructionReadRafWitness};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend, SumcheckKernel,
};

use super::{
    CycleRa, InstructionCycleRow, LookupIndexChunks, LookupIndexRow,
    OptimizedInstructionReadRafKernel, RepeatCache, ScanBlock, SharedInstructionRows,
    StreamingInstructionReadRaf, SCAN_BLOCK,
};

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

fn challenge<F: JoltField>(round: usize) -> F {
    F::from_u64(0x9E37_79B9_7F4A_7C15 ^ (round as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x11)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
    let tables = [
        LookupTableKind::<RISCV_XLEN>::And(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::Andn(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::Or(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::Xor(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::VirtualXORROTW7(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::UnsignedLessThan(Default::default()).index(),
        LookupTableKind::<RISCV_XLEN>::Equal(Default::default()).index(),
    ];
    let mut state = seed;
    (0..1usize << log_t)
        .map(|j| {
            let lookup_index = match j {
                0 => 0u128,
                1 => u128::MAX,
                2 => ((u64::MAX as u128) << 64) | splitmix(&mut state) as u128,
                _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
            };
            let table_index = if j % 7 == 3 {
                None
            } else {
                Some(tables[j % tables.len()])
            };
            InstructionReadRafWitness {
                lookup_index: LookupIndex(lookup_index),
                table_index: TableIndex(table_index),
                raf_flag: InstructionRafFlag(j % 3 == 0),
            }
        })
        .collect()
}

#[test]
fn shared_rows_survive_handoff_then_release() {
    with_sample_backend(|backend| {
        let cycles = backend.random_access().unwrap().cycles();
        let mut session = ProofSession::default();
        let rows =
            InstructionCycleRow::shared_for_next_consumer::<Fr>(&mut session, backend, cycles)
                .unwrap();
        let weak = Arc::downgrade(&rows);
        drop(rows);
        assert!(weak.upgrade().is_some());

        let rows =
            InstructionCycleRow::shared_for_next_consumer::<Fr>(&mut session, backend, cycles)
                .unwrap();
        assert!(Arc::ptr_eq(&weak.upgrade().unwrap(), &rows));
        drop(rows);
        assert!(weak.upgrade().is_some());

        let rows = InstructionCycleRow::shared::<Fr>(&mut session, backend, cycles).unwrap();
        assert!(Arc::ptr_eq(&weak.upgrade().unwrap(), &rows));
        drop(rows);
        assert!(weak.upgrade().is_none());
    });
}

#[test]
fn packed_instruction_row_roundtrips() {
    let lookup_index = u128::MAX - 17;
    let table = LookupTableKind::<RISCV_XLEN>::COUNT - 1;
    let row = InstructionCycleRow::new(
        lookup_index,
        Some(table),
        true,
        u32::MAX as usize,
        Some(u64::MAX - 1),
        #[cfg(feature = "akita")]
        FusedInc(-123),
    );
    assert_eq!(row.lookup_index(), lookup_index);
    assert_eq!(row.table_index(), Some(table));
    assert_eq!(row.bytecode_pc(), u32::MAX as usize);
    assert_eq!(row.remapped_ram_address(), Some(u64::MAX - 1));
    assert!(row.raf_flag());
    #[cfg(feature = "akita")]
    assert_eq!(row.fused_inc::<Fr>(), -Fr::from_u64(123));
}

#[test]
fn packed_lookup_binding_matches_multilinear_weights() {
    fn check<F: JoltField>(columns: usize, bits: usize) {
        use super::super::{constant_suffix::ConstantSuffix, lazy_ra::LazyFoldedRa};
        use rand_chacha::ChaCha20Rng;
        use rand_core::{RngCore, SeedableRng};

        let mut rng = ChaCha20Rng::seed_from_u64(314_159);
        let cycles = 1 << 16;
        let prefix = cycles - 7;
        let indices: Vec<u128> = (0..cycles)
            .map(|i| {
                if i >= prefix {
                    0x1234_5678_9abc_def0_fedc_ba98_7654_3210
                } else {
                    (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())
                }
            })
            .collect();
        let rows = indices
            .iter()
            .map(|&index| {
                InstructionCycleRow::new(
                    index,
                    None,
                    false,
                    0,
                    None,
                    #[cfg(feature = "akita")]
                    FusedInc::default(),
                )
            })
            .collect();
        let source =
            LookupIndexChunks::new(Arc::new(ConstantSuffix::new(rows, cycles)), columns, bits);
        let tables: Vec<Vec<F>> = (0..columns)
            .map(|_| (0..16).map(|_| F::random(&mut rng)).collect())
            .collect();
        let challenges = [3, 5, 13, 29].map(F::from_u64);
        let mut bound = LazyFoldedRa::new(tables.clone(), source);
        for challenge in challenges {
            bound.bind(challenge);
        }
        let weights: Vec<F> = (0..16)
            .map(|branch| {
                challenges
                    .iter()
                    .enumerate()
                    .fold(F::one(), |weight, (bit, r)| {
                        weight
                            * if (branch >> bit) & 1 == 1 {
                                *r
                            } else {
                                F::one() - *r
                            }
                    })
            })
            .collect();
        for (column, table) in tables.iter().enumerate() {
            for row in 0..cycles / 16 {
                let expected: F = weights
                    .iter()
                    .enumerate()
                    .map(|(branch, weight)| {
                        let digit = ((indices[row * 16 + branch]
                            >> (bits * (columns - 1 - column)))
                            & ((1u128 << bits) - 1)) as usize;
                        *weight * table[digit]
                    })
                    .sum();
                assert_eq!(bound.value(column, row), expected);
            }
        }
    }
    check::<Fr>(16, 2);
    check::<Fr>(16, 4);
    check::<Fr>(20, 4);
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>(32, 4);
}

/// The sumcheck input claim from first principles:
/// `Σ_j eq(r_reduction, j) · (Val_j(k_j) + γ·RafVal_j(k_j))` with the
/// point-mass `ra` collapsed at each cycle's lookup index. Pins both
/// kernels to the protocol, not merely to each other (each kernel's own
/// `s(0) + s(1) = claim` self-check would reject a drifted round 0).
fn input_claim<F: JoltField>(rows: &[InstructionReadRafWitness], r_reduction: &[F], gamma: F) -> F {
    let tables: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::iter().collect();
    let gamma_sqr = gamma * gamma;
    let address_bits = 2 * RISCV_XLEN;
    eq_table(r_reduction)
        .iter()
        .zip(rows)
        .map(|(&u, row)| {
            let k = row.lookup_index.0;
            let value = row.table_index.0.map_or_else(
                || F::from_u64(0),
                |index| F::from_u64(tables[index].materialize_entry(k)),
            );
            let raf = if !row.raf_flag.0 {
                let (left, right) = LookupBits::new(k, address_bits).uninterleave();
                gamma * F::from_u64(u64::from(left)) + gamma_sqr * F::from_u64(u64::from(right))
            } else {
                let mut raf = gamma_sqr
                    * (F::from_u64(k as u64) + F::from_u64((k >> 64) as u64).mul_pow_2(64));
                if CANONICAL_INSTRUCTION_ADDRESS
                    && (k >> (address_bits / 2)) == (1u128 << (address_bits / 2)) - 1
                {
                    raf += gamma_sqr * gamma;
                }
                raf
            };
            u * (value + raf)
        })
        .sum()
}

fn assert_parity(log_t: usize, num_virtual_ra_polys: usize, seed: u64) {
    assert_parity_case(log_t, num_virtual_ra_polys, seed, None);
}

fn assert_parity_case(
    log_t: usize,
    num_virtual_ra_polys: usize,
    seed: u64,
    exceptional: Option<ExceptionalEq>,
) {
    assert_parity_prefix(log_t, num_virtual_ra_polys, seed, exceptional, None);
}

#[test]
fn constant_suffix_rounds_match_reference() {
    for prefix in [0, 1, 2, 15, 16, 17, 33, 63] {
        for exceptional in [None].into_iter().chain(ExceptionalEq::ALL.map(Some)) {
            for zero_tail in [true, false] {
                assert_parity_prefix(6, 8, 123, exceptional, Some((prefix, zero_tail)));
            }
        }
    }
}

fn assert_parity_prefix(
    log_t: usize,
    num_virtual_ra_polys: usize,
    seed: u64,
    exceptional: Option<ExceptionalEq>,
    prefix: Option<(usize, bool)>,
) {
    assert_parity_prefix_field::<Fr>(log_t, num_virtual_ra_polys, seed, exceptional, prefix);
    #[cfg(feature = "akita")]
    assert_parity_prefix_field::<Prime128Offset6A23801F>(
        log_t,
        num_virtual_ra_polys,
        seed,
        exceptional,
        prefix,
    );
}

fn assert_parity_prefix_field<F: JoltField>(
    log_t: usize,
    num_virtual_ra_polys: usize,
    seed: u64,
    exceptional: Option<ExceptionalEq>,
    prefix: Option<(usize, bool)>,
) {
    let dimensions = InstructionReadRafDimensions::new(
        log_t,
        2 * RISCV_XLEN,
        NonZeroUsize::new(num_virtual_ra_polys).unwrap(),
    );
    let mut rows = fixture_rows(log_t, seed);
    if let Some((prefix, zero_tail)) = prefix {
        let mut tail = rows[rows.len() - 1];
        if zero_tail {
            tail.lookup_index = LookupIndex(0);
            tail.table_index = TableIndex(None);
            tail.raf_flag = InstructionRafFlag(false);
        }
        for row in &mut rows[prefix..] {
            *row = tail;
        }
    }
    let r_reduction: Vec<F> = exceptional.map_or_else(
        || {
            (0..log_t)
                .map(|i| F::from_u64(1000 + 37 * i as u64))
                .collect()
        },
        |case| case.point(log_t, challenge(dimensions.instruction_address_bits())),
    );
    assert_parity_rows(dimensions, &r_reduction, rows);
}

fn assert_parity_rows<F: JoltField>(
    dimensions: InstructionReadRafDimensions,
    r_reduction: &[F],
    rows: Vec<InstructionReadRafWitness>,
) {
    assert_parity_rows_with_challenges(dimensions, r_reduction, rows, challenge);
}

/// The streaming kernel over `rows`, retained as the two typed views its
/// slot takes of a witness's resident rows.
fn streamed<F: JoltField>(
    dimensions: InstructionReadRafDimensions,
    r_reduction: &[F],
    rows: &[InstructionReadRafWitness],
    gamma: F,
) -> Result<OptimizedInstructionReadRafKernel<F>, KernelError<F>> {
    let lookup_indices = rows
        .iter()
        .map(|row| LookupIndexRow {
            lookup_index: row.lookup_index,
        })
        .collect();
    OptimizedInstructionReadRafKernel::streamed(
        dimensions,
        r_reduction,
        BundleStore::Retained(rows.to_vec()),
        BundleStore::Retained(lookup_indices),
        gamma,
    )
}

/// Drives the default and the streaming kernel beside the reference: every
/// round polynomial and the output claims of both equal the reference's.
fn assert_parity_rows_with_challenges<F: JoltField>(
    dimensions: InstructionReadRafDimensions,
    r_reduction: &[F],
    rows: Vec<InstructionReadRafWitness>,
    round_challenge: impl Fn(usize) -> F,
) {
    let gamma = F::from_u64(0xACE1_57EF);

    let mut reference =
        InstructionReadRafKernel::new(dimensions, r_reduction, rows.clone(), gamma).unwrap();
    let mut kernels = [
        (
            "default",
            OptimizedInstructionReadRafKernel::new(
                dimensions,
                r_reduction,
                Arc::new(pack(&rows).into()),
                gamma,
            )
            .unwrap(),
        ),
        (
            "streaming",
            streamed(dimensions, r_reduction, &rows, gamma).unwrap(),
        ),
    ];

    let rounds = reference.num_rounds();
    let mut claim = input_claim(&rows, r_reduction, gamma);
    for round in 0..rounds {
        let bind = round.checked_sub(1).map(&round_challenge);
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        for (slot, kernel) in &mut kernels {
            assert_eq!(rounds, kernel.num_rounds());
            let poly = kernel.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                poly.coefficients(),
                "{slot} round {round} polynomial mismatch"
            );
        }
        claim = reference_poly.evaluate(round_challenge(round));
    }
    reference
        .finish_rounds(round_challenge(rounds - 1))
        .unwrap();

    let inputs = InstructionReadRafInputClaims {
        lookup_output: F::from_u64(0),
        left_lookup_operand: F::from_u64(0),
        right_lookup_operand: F::from_u64(0),
    };
    let reference_outputs = reference.output_claims(&inputs).unwrap();
    for (slot, mut kernel) in kernels {
        kernel.finish_rounds(round_challenge(rounds - 1)).unwrap();
        let outputs = kernel.output_claims(&inputs).unwrap();
        assert_eq!(
            reference_outputs.lookup_table_flags, outputs.lookup_table_flags,
            "{slot}"
        );
        assert_eq!(
            reference_outputs.instruction_ra, outputs.instruction_ra,
            "{slot}"
        );
        assert_eq!(
            reference_outputs.instruction_raf_flag, outputs.instruction_raf_flag,
            "{slot}"
        );
    }
}

fn check_repeated_lookup_rows<F: JoltField>(log_t: usize, prefix: usize, constant: Option<F>) {
    let dimensions =
        InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
    let mut pattern = fixture_rows(5, 97531);
    pattern[1].lookup_index = pattern[0].lookup_index;
    pattern[2].lookup_index = pattern[0].lookup_index;
    pattern[2].table_index = pattern[0].table_index;
    pattern[2].raf_flag = InstructionRafFlag(!pattern[0].raf_flag.0);
    let rows: Vec<_> = (0..1 << log_t)
        .map(|i| pattern[if i < prefix { i % pattern.len() } else { 0 }])
        .collect();
    assert!(ScanBlock::should_coalesce(&pack(&rows)));
    let point: Vec<F> = (0..log_t)
        .map(|i| constant.unwrap_or_else(|| challenge(i)))
        .collect();
    assert_parity_rows(dimensions, &point, rows);
}

/// Range-check rows whose keys widen from block to block: 32 bits, 64 bits,
/// 72 bits, then the full address.
fn range_check_rows(log_t: usize, repeated: bool) -> Vec<InstructionReadRafWitness> {
    let table = LookupTableKind::<RISCV_XLEN>::RangeCheck(Default::default()).index();
    (0..1 << log_t)
        .map(|i| {
            let value = if repeated { i % 32 } else { i } as u128 + 1;
            let low = value.wrapping_mul(0x9e37_79b9_7f4a_7c15) & u128::from(u64::MAX);
            let index = match i / SCAN_BLOCK {
                0 => low & u128::from(u32::MAX),
                1 => low,
                2 => (0xffu128 << 64) | low,
                _ => (1u128 << 127) | low,
            };
            InstructionReadRafWitness {
                lookup_index: LookupIndex(index),
                table_index: TableIndex(if i % 7 == 0 { None } else { Some(table) }),
                raf_flag: InstructionRafFlag(i % 3 != 0),
            }
        })
        .collect()
}

#[test]
fn range_prefix_caches_match_reference_through_boundaries() {
    fn check<F: JoltField>() {
        let log_t = 12;
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let point: Vec<F> = (0..log_t).map(challenge).collect();
        for (repeated, zero_prefix_factor) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let rows = range_check_rows(log_t, repeated);
            assert_eq!(ScanBlock::should_coalesce(&pack(&rows)), repeated);
            assert_parity_rows_with_challenges(dimensions, &point, rows, |round| {
                if zero_prefix_factor && round == 0 {
                    F::one()
                } else {
                    challenge(round)
                }
            });
        }
    }
    check::<Fr>();
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>();
}

#[test]
fn zero_range_keys_match_reference_through_final_address_phase() {
    fn check<F: JoltField>() {
        let log_t = 5;
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let table = LookupTableKind::<RISCV_XLEN>::RangeCheck(Default::default()).index();
        let point: Vec<F> = (0..log_t).map(challenge).collect();
        for prefix in [0, 7] {
            let rows = (0..1 << log_t)
                .map(|i| InstructionReadRafWitness {
                    lookup_index: LookupIndex(0),
                    table_index: TableIndex(Some(table)),
                    raf_flag: InstructionRafFlag(i >= prefix || i % 2 == 0),
                })
                .collect();
            assert_parity_rows(dimensions, &point, rows);
        }
    }
    check::<Fr>();
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>();
}

#[test]
fn coalescing_preserves_weight_for_colliding_keys_and_classes() {
    fn check<F: JoltField>() {
        // Four slots for up to ten distinct (key, class) pairs force collisions.
        let mut cache = RepeatCache::new(1);
        for length in [64, 17] {
            let rows: Vec<_> = (0..length)
                .map(|i| {
                    let high = [0u64, 0, 1, 2, 0, 3, 3, 4][i % 8];
                    let key = (u128::from(high) << 64) | u128::from(high.rotate_left(32));
                    InstructionCycleRow::new(
                        key,
                        None,
                        i >= length / 2,
                        0,
                        None,
                        #[cfg(feature = "akita")]
                        FusedInc::default(),
                    )
                })
                .collect();
            let original: Vec<F> = (0..length)
                .map(|i| F::from_u64((i * i + 1) as u64))
                .collect();
            let mut weights = original.clone();
            let block = ScanBlock::coalesced(&rows, &mut weights, &mut cache);
            for (class, range) in block.starts.windows(2).enumerate() {
                for high in 0..5u64 {
                    let key = (u128::from(high) << 64) | u128::from(high.rotate_left(32));
                    let expected: F = rows
                        .iter()
                        .zip(&original)
                        .filter(|(row, _)| {
                            ScanBlock::class(*row) == class && row.lookup_index() == key
                        })
                        .map(|(_, weight)| *weight)
                        .sum();
                    let actual: F = block.offsets[usize::from(range[0])..usize::from(range[1])]
                        .iter()
                        .map(|offset| usize::from(*offset))
                        .filter(|offset| rows[*offset].lookup_index() == key)
                        .map(|offset| weights[offset])
                        .sum();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    check::<Fr>();
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>();
}

#[test]
fn repeated_lookup_rows_match_reference() {
    fn check<F: JoltField>() {
        for prefix in [1027, 1 << 11] {
            for constant in [None, Some(F::zero()), Some(F::one())] {
                check_repeated_lookup_rows(11, prefix, constant);
            }
        }
    }
    check::<Fr>();
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>();
}

#[test]
fn repeated_lookup_rows_match_reference_across_scan_tasks() {
    for prefix in [6003, 1 << 14] {
        check_repeated_lookup_rows::<Fr>(14, prefix, None);
        #[cfg(feature = "akita")]
        check_repeated_lookup_rows::<Prime128Offset6A23801F>(14, prefix, None);
    }
}

#[test]
fn parity_default_geometry() {
    assert_parity(4, 8, 12345);
}

#[test]
fn parity_wide_virtual_chunks_and_odd_log_t() {
    assert_parity(3, 4, 67890);
}

/// Enough cycles for several scan tasks, so the phase sums merge tasks
/// that accumulated the same tables.
#[test]
fn parity_multi_task_scan() {
    assert_parity(11, 8, 24680);
}

/// 8-, 32- and 64-bit virtual chunks across the third cycle bind's dense
/// switch: single 8-bit columns, and products of two and four 16-bit
/// columns.
#[test]
fn parity_virtual_chunk_widths_across_dense_switch() {
    for (num_virtual_ra_polys, seed) in [(16, 11), (4, 22), (2, 33)] {
        assert_parity(6, num_virtual_ra_polys, seed);
    }
}

#[test]
fn parity_all_raf_rows() {
    let log_t = 3;
    let dimensions =
        InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
    let rows: Vec<InstructionReadRafWitness> = fixture_rows(log_t, 555)
        .into_iter()
        .map(|mut row| {
            row.raf_flag = InstructionRafFlag(true);
            row
        })
        .collect();
    let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(2000 + 11 * i as u64)).collect();
    let gamma = fr(0xBEEF);

    let mut reference =
        InstructionReadRafKernel::new(dimensions, &r_reduction, rows.clone(), gamma).unwrap();
    let mut optimized = OptimizedInstructionReadRafKernel::new(
        dimensions,
        &r_reduction,
        Arc::new(pack(&rows).into()),
        gamma,
    )
    .unwrap();
    let mut claim = input_claim(&rows, &r_reduction, gamma);
    for round in 0..reference.num_rounds() {
        let bind = round.checked_sub(1).map(challenge);
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            reference_poly.coefficients(),
            optimized_poly.coefficients(),
            "round {round}"
        );
        claim = reference_poly.evaluate(challenge(round));
    }
}
#[test]
fn parity_exceptional_eq_in_lazy_and_dense_cycle_tables() {
    for virtuals in [4usize, 8] {
        for case in ExceptionalEq::ALL {
            assert_parity_case(6, virtuals, 257, Some(case));
        }
    }
}

/// The streaming kernel against the default kernel on domains where the
/// virtual RA polynomials stay streamed past the first cycle bind. Sixteen
/// 8-bit columns materialize at the second bind of 2^11 cycles and at the
/// third bind of 2^13 (both fused) and of 2^15 (after doubling); at 2^6
/// cycles, as in every 16-bit case of the reference parity tests, at the
/// first. Each domain ends in its last row alone and in constant suffixes
/// that start after one row and two, inside a pair, at five eighths of the
/// domain and, where it is that long, at a visit callback's end.
#[test]
fn streaming_cycle_rounds_match_default_at_every_materializing_bind() {
    let cases =
        [(6, 1), (11, 2), (13, 3), (15, 3)]
            .into_iter()
            .flat_map(|(log_t, materializing_bind)| {
                let cycles = 1usize << log_t;
                [cycles, 1, 2, 37, 5 * cycles / 8, VISIT_ROWS]
                    .into_iter()
                    .filter(move |prefix| *prefix <= cycles)
                    .map(move |prefix| (log_t, materializing_bind, prefix))
            });
    for (log_t, materializing_bind, prefix) in cases {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            2 * RISCV_XLEN,
            NonZeroUsize::new(16).unwrap(),
        );
        let mut rows = fixture_rows(log_t, 97 + log_t as u64);
        let tail = rows[rows.len() - 1];
        rows[prefix..].fill(tail);
        let r_reduction: Vec<Fr> = (0..log_t).map(challenge).collect();
        let gamma = fr(0xACE1_57EF);
        let mut default = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            Arc::new(pack(&rows).into()),
            gamma,
        )
        .unwrap();
        let mut streaming = streamed(dimensions, &r_reduction, &rows, gamma).unwrap();
        let rounds = default.num_rounds();
        let mut claim = input_claim(&rows, &r_reduction, gamma);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let expected = default.prove_round(bind, round, claim).unwrap();
            let actual = streaming.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                expected.coefficients(),
                actual.coefficients(),
                "log_t {log_t}, prefix {prefix}, round {round}"
            );
            claim = expected.evaluate(challenge(round));
            if let Some(cycle) = &streaming.cycle {
                let cycle_binds = round - dimensions.instruction_address_bits();
                assert_eq!(
                    matches!(cycle.ra, CycleRa::Streamed(_)),
                    cycle_binds < materializing_bind,
                    "log_t {log_t}, prefix {prefix}, cycle bind {cycle_binds}"
                );
            }
        }
        default.finish_rounds(challenge(rounds - 1)).unwrap();
        streaming.finish_rounds(challenge(rounds - 1)).unwrap();
        let inputs = InstructionReadRafInputClaims {
            lookup_output: fr(0),
            left_lookup_operand: fr(0),
            right_lookup_operand: fr(0),
        };
        let expected = default.output_claims(&inputs).unwrap();
        let actual = streaming.output_claims(&inputs).unwrap();
        assert_eq!(expected.lookup_table_flags, actual.lookup_table_flags);
        assert_eq!(expected.instruction_ra, actual.instruction_ra);
        assert_eq!(expected.instruction_raf_flag, actual.instruction_raf_flag);
    }
}

/// Rows whose lookup index turns constant at cycle 20 and whose table index
/// and RAF flag at cycle 41: the streamed cycle round reads the combined
/// value by pair row, so the virtual RA family's suffix has to start where
/// the claim column's does, not where the lookup indices' does.
#[test]
fn streamed_suffix_starts_where_the_claim_column_turns_constant() {
    let log_t = 6;
    let dimensions =
        InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
    let mut rows = fixture_rows(log_t, 4242);
    let tail = rows[rows.len() - 1];
    for row in &mut rows[20..41] {
        row.lookup_index = tail.lookup_index;
    }
    rows[41..].fill(tail);
    let point: Vec<Fr> = (0..log_t).map(challenge).collect();
    assert_parity_rows(dimensions, &point, rows);
}

/// On one pool thread a scan task of 2^14 rows holds two blocks and a
/// claim-byte task four visit callbacks, so the resident passes place each
/// block's weights and range cache, and each callback's claim bytes, by its
/// offset in the task. The first three blocks hold range-check keys of
/// different widths, whose caches differ.
#[test]
fn resident_tasks_of_several_blocks_match_reference() {
    let check = || {
        let log_t = 14;
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let point: Vec<Fr> = (0..log_t).map(challenge).collect();
        assert_parity_rows(dimensions, &point, range_check_rows(log_t, false));
    };
    #[cfg(feature = "parallel")]
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(check);
    #[cfg(not(feature = "parallel"))]
    check();
}

/// The streaming constructor refuses a row whose table index names no lookup
/// table in its first pass, before a claim byte packs it or a scan buckets
/// it.
#[test]
fn streaming_kernel_refuses_an_unknown_lookup_table() {
    let dimensions =
        InstructionReadRafDimensions::new(2, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
    let mut rows = fixture_rows(2, 5);
    rows[1].table_index = TableIndex(Some(usize::from(u8::MAX)));
    assert!(matches!(
        streamed(dimensions, &[fr(3), fr(5)], &rows, fr(7)),
        Err(KernelError::InvariantViolation { .. })
    ));
}

/// The relation and prover inputs of a `2^log_t`-cycle stage with eight
/// 16-bit virtual RA polynomials.
struct TraceCase<F: JoltField> {
    relation: InstructionReadRaf<F>,
    claims: InstructionReadRafInputClaims<F>,
    points: InstructionReadRafInputClaims<Vec<F>>,
    challenges: InstructionReadRafChallenges<F>,
}

impl<F: JoltField> TraceCase<F> {
    fn new(log_t: usize) -> Self {
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        Self {
            relation: InstructionReadRaf::new(dimensions),
            claims: InstructionReadRafInputClaims {
                lookup_output: F::zero(),
                left_lookup_operand: F::zero(),
                right_lookup_operand: F::zero(),
            },
            points: InstructionReadRafInputClaims {
                lookup_output: (0..log_t)
                    .map(|i| F::from_u64(7000 + 29 * i as u64))
                    .collect(),
                left_lookup_operand: Vec::new(),
                right_lookup_operand: Vec::new(),
            },
            challenges: InstructionReadRafChallenges {
                gamma: F::from_u64(0xFEED_5EED),
            },
        }
    }

    fn inputs(&self) -> ProverInputs<'_, F, InstructionReadRaf<F>> {
        ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.challenges,
        }
    }
}

/// Every round and the output claims through the streaming slot over a
/// witness's resident rows, against the reference, in both fields. The traces
/// fill their domain, end one cycle short of it, hold one cycle or two, and
/// end inside a pair and a split-eq group or on a pair boundary. A one-cycle
/// domain has no cycle round: the handoff gathers the RA claims at once.
/// The slot parks no row cache in the session.
#[test]
fn streaming_slot_matches_reference_over_a_trace_witness() {
    fn check<F: JoltField>(backend: &TraceBackend, log_t: usize) {
        let case = TraceCase::<F>::new(log_t);
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, case.inputs())
            .unwrap();
        let mut session = ProofSession::default();
        let mut streaming = StreamingInstructionReadRaf
            .prepare(&mut session, backend, case.inputs())
            .unwrap();
        assert!(session.state::<SharedInstructionRows>().is_none());
        let input_claim = probe_input_claim(reference.as_mut());
        let challenges: Vec<F> = (0..reference.num_rounds()).map(challenge).collect();
        run_lockstep(
            reference.as_mut(),
            streaming.as_mut(),
            input_claim,
            &challenges,
        );
        let expected = reference.output_claims(&case.claims).unwrap();
        let actual = streaming.output_claims(&case.claims).unwrap();
        assert_eq!(expected.lookup_table_flags, actual.lookup_table_flags);
        assert_eq!(expected.instruction_ra, actual.instruction_ra);
        assert_eq!(expected.instruction_raf_flag, actual.instruction_raf_flag);
    }
    for (log_t, cycles) in [
        (0, 1),
        (3, 5),
        (3, 8),
        (6, 1),
        (6, 2),
        (6, 37),
        (6, 40),
        (6, 63),
        (6, 64),
    ] {
        structured_fixture(cycles).with_plane(log_t, |backend| {
            check::<Fr>(backend, log_t);
            #[cfg(feature = "akita")]
            check::<Prime128Offset6A23801F>(backend, log_t);
        });
    }
}

/// `with_streaming` puts this kernel in the backend's slot. Streamed and
/// default proofs are byte-equal, so only the kernel's behavior shows which
/// one is installed: it refuses a witness without resident rows, which the
/// default kernel collects. Dory stands in for the commitment scheme; the
/// akita build's `optimized` takes another.
#[cfg(not(feature = "akita"))]
#[test]
fn with_streaming_installs_the_instruction_read_raf_kernel() {
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
                .instruction_read_raf
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
