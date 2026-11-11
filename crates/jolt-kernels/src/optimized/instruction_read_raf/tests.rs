use crate::optimized::parity::ExceptionalEq;
use std::num::NonZeroUsize;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Fr, JoltField, Ring};
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_sumcheck::ProveRounds;
use jolt_witness::testing::with_sample_backend;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
use jolt_witness::RowSource;

use crate::reference::instruction_read_raf::{InstructionReadRafKernel, InstructionReadRafWitness};
use crate::reference::views::eq_table;
use crate::{ProofSession, SumcheckKernel};

use super::{
    InstructionCycleRow, LookupIndexChunks, OptimizedInstructionReadRafKernel, RepeatCache,
    ScanBlock, SCAN_BLOCK,
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
    for prefix in [0, 1, 15, 16, 17, 33, 63] {
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

fn assert_parity_rows_with_challenges<F: JoltField>(
    dimensions: InstructionReadRafDimensions,
    r_reduction: &[F],
    rows: Vec<InstructionReadRafWitness>,
    round_challenge: impl Fn(usize) -> F,
) {
    let gamma = F::from_u64(0xACE1_57EF);

    let mut reference =
        InstructionReadRafKernel::new(dimensions, r_reduction, rows.clone(), gamma).unwrap();
    let mut optimized = OptimizedInstructionReadRafKernel::new(
        dimensions,
        r_reduction,
        Arc::new(pack(&rows).into()),
        gamma,
    )
    .unwrap();

    let rounds = reference.num_rounds();
    assert_eq!(rounds, optimized.num_rounds());
    let mut claim = input_claim(&rows, r_reduction, gamma);
    for round in 0..rounds {
        let bind = round.checked_sub(1).map(&round_challenge);
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            reference_poly.coefficients(),
            optimized_poly.coefficients(),
            "round {round} polynomial mismatch"
        );
        claim = reference_poly.evaluate(round_challenge(round));
    }
    reference
        .finish_rounds(round_challenge(rounds - 1))
        .unwrap();
    optimized
        .finish_rounds(round_challenge(rounds - 1))
        .unwrap();

    let inputs = InstructionReadRafInputClaims {
        lookup_output: F::from_u64(0),
        left_lookup_operand: F::from_u64(0),
        right_lookup_operand: F::from_u64(0),
    };
    let reference_outputs = reference.output_claims(&inputs).unwrap();
    let optimized_outputs = optimized.output_claims(&inputs).unwrap();
    assert_eq!(
        reference_outputs.lookup_table_flags,
        optimized_outputs.lookup_table_flags
    );
    assert_eq!(
        reference_outputs.instruction_ra,
        optimized_outputs.instruction_ra
    );
    assert_eq!(
        reference_outputs.instruction_raf_flag,
        optimized_outputs.instruction_raf_flag
    );
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

#[test]
fn range_prefix_caches_match_reference_through_boundaries() {
    fn check<F: JoltField>() {
        let log_t = 12;
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let table = LookupTableKind::<RISCV_XLEN>::RangeCheck(Default::default()).index();
        let point: Vec<F> = (0..log_t).map(challenge).collect();
        for (repeated, zero_prefix_factor) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let rows: Vec<_> = (0..1 << log_t)
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
                .collect();
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
                            ScanBlock::class(row) == class && row.lookup_index() == key
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
