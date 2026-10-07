#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests assert valid kernel geometry and explicit rejection"
)]

use super::*;
use std::{ops::Range, sync::Arc};

use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_params::{BasisMode, SetupMatrixCapacity, SubringCoefficientPackingGeometry};
use akita_pcs::custom_source::{
    CommitInnerPlan, CpuFoldResponses, DecomposeFoldBatchPlan, OneHotBatchView, OpeningBatchKernel,
    RootOpeningSource, RootPolyShape, SourceCoefficients, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPlan,
};
use akita_pcs::AkitaError;
use akita_pcs::{AkitaProverSetup, CpuBackend, OneHotPoly};
use akita_types::PreparedSubringCoefficientPackingPoint;
use jolt_field::{Fp128x8i32, One, Ring};

use super::commit::commit_columns;
use super::digit_windows::{flush_digit_accumulators, DigitWindows};
use super::source::TraceOneHotColumnBatchView;
use crate::{AkitaField, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams};
use jolt_openings::CommitmentScheme;
use jolt_poly::OneHotPolynomial;

#[derive(Debug)]
struct TestRows {
    rows: usize,
    columns: usize,
    k: usize,
    committed_zero_column: Option<usize>,
}

impl TestRows {
    fn selected_row(&self, row: usize, column: usize) -> u8 {
        ((row * (2 * column + 1) + column) % self.k) as u8
    }
}

impl TraceOneHotRows for TestRows {
    fn num_rows(&self) -> usize {
        self.rows
    }

    fn num_columns(&self) -> usize {
        self.columns
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        for (column, selected) in selected_rows.iter_mut().enumerate() {
            *selected = self.selected_row(row, column);
        }
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        self.committed_zero_column
            .filter(|&column| self.selected_row(row, column) == 0)
            .map_or(0, |column| 1u64 << column)
    }
}

type TestBackend = CpuBackend<AkitaField, AkitaField>;

/// The kernels under test never read the owned setup; the smallest valid
/// setup only gives them a backend to hang off.
fn test_backend() -> TestBackend {
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        1,
        1,
        SetupMatrixCapacity {
            num_field_elements: 1,
        },
    )
    .unwrap();
    CpuBackend::new(setup.expanded).unwrap()
}

fn packing_point<const D: usize>(
    source_num_vars: usize,
    num_live_positions: usize,
    num_positions_per_block: usize,
) -> PreparedSubringCoefficientPackingPoint<AkitaField> {
    let geometry = SubringCoefficientPackingGeometry::try_new(1, D, 64).unwrap();
    let point = (0..source_num_vars)
        .map(|index| AkitaField::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        num_live_positions,
        num_positions_per_block,
        source_num_vars,
        &point,
    )
    .unwrap()
}

fn assert_ring_mapping<const D: usize>(
    k: usize,
    rows: usize,
    committed_zero_column: Option<usize>,
) {
    let columns = TraceOneHotColumn::new(
        k,
        64,
        Arc::new(TestRows {
            rows,
            columns: 3,
            k,
            committed_zero_column,
        }),
    )
    .unwrap();
    let source = &columns[0];
    let segment_rings = source.segment_ring_elems::<D>().unwrap();
    let mut actual = Vec::new();
    visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
        actual.extend(
            contributions
                .iter()
                .map(|&(column, coefficient)| (column, ring * D + coefficient)),
        );
    })
    .unwrap();
    let expected = (0..rows)
        .flat_map(|row| {
            (0..3).filter_map(move |column| {
                let selected_row = (row * (2 * column + 1) + column) % k;
                (selected_row != 0 || committed_zero_column == Some(column))
                    .then_some((column, row * k + selected_row))
            })
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = expected;
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

#[test]
fn row_major_mapping_is_dimension_generic() {
    for rows in [32, 64] {
        assert_ring_mapping::<64>(16, rows, None);
        assert_ring_mapping::<128>(16, rows, None);
        assert_ring_mapping::<256>(16, rows, None);
        assert_ring_mapping::<512>(16, rows, None);
        assert_ring_mapping::<64>(256, rows, None);
        assert_ring_mapping::<128>(256, rows, None);
        assert_ring_mapping::<256>(256, rows, None);
        assert_ring_mapping::<512>(256, rows, None);
    }
}

#[test]
fn committed_digit_zero_mapping_is_dimension_generic() {
    assert_ring_mapping::<64>(16, 32, Some(1));
    assert_ring_mapping::<64>(256, 32, Some(0));
}

fn digit_window_source<const D: usize>() -> CyclotomicRing<AkitaField, D> {
    // Mix small, negative, and near-modulus coefficients so both window
    // halves carry dense 16-bit digits.
    CyclotomicRing::from_coefficients(std::array::from_fn(|index| match index % 3 {
        0 => AkitaField::from_u64((index + 1) as u64),
        1 => -AkitaField::from_u64((index + 1) as u64),
        _ => AkitaField::from_u128(u128::MAX / (index as u128 + 2)),
    }))
}

fn assert_digit_windows_match_shift_accumulation<const D: usize>() {
    const COLUMNS: usize = 6;
    let rows_per_ring = D / 16;
    let mut selected_rows = vec![NO_SELECTED_ROW; rows_per_ring * COLUMNS];
    let mut committed_zero_masks = vec![0u64; rows_per_ring];
    for row in 0..rows_per_ring {
        let shared_hot = ((row + 1) % 15 + 1) as u8;
        selected_rows[row * COLUMNS] = shared_hot;
        selected_rows[row * COLUMNS + 1] = shared_hot;
        selected_rows[row * COLUMNS + 2] = ((2 * row + 3) % 15 + 1) as u8;
        selected_rows[row * COLUMNS + 3] = if row % 3 == 1 {
            NO_SELECTED_ROW
        } else {
            ((3 * row + 5) % 15 + 1) as u8
        };
        if row % 2 == 0 {
            committed_zero_masks[row] |= 1 << 4;
        }
    }

    let source = digit_window_source::<D>();
    let mut windows = DigitWindows::<D>::new();
    windows.load(&source);
    let mut actual = vec![[Fp128x8i32([0; 8]); D]; COLUMNS];
    let wide_source: AkitaWideRing<D> = AkitaWideRing::from_ring(&source);
    let mut expected = vec![AkitaWideRing::zero(); COLUMNS];
    for (column, actual) in actual.iter_mut().enumerate() {
        let mut shifts = Vec::new();
        for (row, (row_indices, &mask)) in selected_rows
            .chunks_exact(COLUMNS)
            .zip(&committed_zero_masks)
            .enumerate()
        {
            let hot = row_indices[column];
            if traversal::row_is_committed(hot, mask, column) {
                shifts.push(16 * row + usize::from(hot));
                wide_source
                    .shift_accumulate_into(&mut expected[column], 16 * row + usize::from(hot));
            }
        }
        windows.accumulate(actual, &shifts);
    }

    let empty_column = actual.last_mut().unwrap();
    *empty_column = [Fp128x8i32([7; 8]); D];
    windows.accumulate(empty_column, &[]);
    assert_eq!(*empty_column, [Fp128x8i32([7; 8]); D]);
    *empty_column = [Fp128x8i32([0; 8]); D];
    let mut reduced = vec![CyclotomicRing::zero(); COLUMNS];
    flush_digit_accumulators(&mut actual, &mut reduced);
    let expected = expected
        .into_iter()
        .map(|value| value.reduce::<AkitaField>())
        .collect::<Vec<_>>();
    assert_eq!(reduced, expected);
}

#[test]
fn digit_windows_match_shift_accumulation() {
    assert_digit_windows_match_shift_accumulation::<64>();
    assert_digit_windows_match_shift_accumulation::<128>();
    assert_digit_windows_match_shift_accumulation::<256>();
    assert_digit_windows_match_shift_accumulation::<512>();
}

#[test]
fn digit_windows_stay_exact_at_accumulation_budget() {
    const D: usize = 64;
    let source = CyclotomicRing::<AkitaField, D>::from_coefficients(std::array::from_fn(|index| {
        -AkitaField::from_u64(index as u64 + 1)
    }));
    let mut windows = DigitWindows::<D>::new();
    windows.load(&source);
    let shifts = (0..D).collect::<Vec<_>>();
    let mut actual = [[Fp128x8i32([0; 8]); D]];
    let wide_source: AkitaWideRing<D> = AkitaWideRing::from_ring(&source);
    let mut expected = AkitaWideRing::zero();
    for _ in 0..MAX_WIDE_ACCUMULATIONS / D {
        windows.accumulate(&mut actual[0], &shifts);
        for &shift in &shifts {
            wide_source.shift_accumulate_into(&mut expected, shift);
        }
    }
    let mut reduced = [CyclotomicRing::zero()];
    flush_digit_accumulators(&mut actual, &mut reduced);
    assert_eq!(reduced[0], expected.reduce::<AkitaField>());
}

#[test]
fn batch_enforces_shared_owner_dimensions_and_order() {
    let make = || {
        TraceOneHotColumn::new(
            16,
            64,
            Arc::new(TestRows {
                rows: 32,
                columns: 3,
                k: 16,
                committed_zero_column: None,
            }),
        )
        .unwrap()
    };
    let columns = make();
    let other = make();
    let ordered = columns.iter().collect::<Vec<_>>();
    assert!(source::validate_batch(&ordered).is_ok());
    assert!(source::validate_batch(&[&columns[1], &columns[0], &columns[2]]).is_err());
    assert!(source::validate_batch(&[&columns[0], &other[1], &columns[2]]).is_err());
    let mut wrong = columns[1].clone();
    wrong.num_vars += 1;
    assert!(source::validate_batch(&[&columns[0], &wrong, &columns[2]]).is_err());
    assert!(columns[0].source_coefficients().is_err());
}

fn assert_deferred_fp128_shift_accumulator<const D: usize>() {
    let source: CyclotomicRing<AkitaField, D> =
        CyclotomicRing::from_coefficients(std::array::from_fn(|_| -AkitaField::one()));
    let mut expected: CyclotomicRing<AkitaField, D> = CyclotomicRing::zero();
    let mut deferred: DeferredFp128Ring<D> = DeferredFp128Ring::zero();

    for _ in 0..K256_ROW_BATCH {
        source.shift_accumulate_into(&mut expected, D / 2);
        deferred.shift_accumulate(&source, D / 2);
    }

    assert!(deferred
        .wraps
        .iter()
        .all(|wraps| usize::from(wraps.unsigned_abs()) <= K256_ROW_BATCH));
    assert_eq!(deferred.reduce_and_clear(), expected);
    assert!(deferred.lo.iter().all(|&limb| limb == 0));
    assert!(deferred.hi.iter().all(|&limb| limb == 0));
    assert!(deferred.wraps.iter().all(|&wraps| wraps == 0));

    let mut expected_after_reuse = CyclotomicRing::zero();
    source.shift_accumulate_into(&mut expected_after_reuse, D - 1);
    deferred.shift_accumulate(&source, D - 1);
    assert_eq!(deferred.reduce_and_clear(), expected_after_reuse);
    assert_eq!(std::mem::size_of::<DeferredFp128Ring<D>>(), 18 * D);
}

#[test]
fn deferred_fp128_shift_accumulator_matches_canonical_at_batch_bound() {
    assert_deferred_fp128_shift_accumulator::<64>();
    assert_deferred_fp128_shift_accumulator::<128>();
    assert_deferred_fp128_shift_accumulator::<256>();
}

fn assert_production_kernels_match_materialized<const D: usize>(
    k: usize,
    rows: usize,
    num_positions: usize,
    committed_zero_column: Option<usize>,
    num_digits: usize,
) {
    const COLUMNS: usize = 3;
    let columns = TraceOneHotColumn::new(
        k,
        64,
        Arc::new(TestRows {
            rows,
            columns: COLUMNS,
            k,
            committed_zero_column,
        }),
    )
    .unwrap();
    let source = &columns[0];
    let materialized_columns = (0..COLUMNS)
        .map(|column| {
            let indices = (0..rows)
                .map(|row| {
                    let hot = ((row * (2 * column + 1) + column) % k) as u8;
                    (hot != 0 || committed_zero_column == Some(column)).then_some(hot)
                })
                .collect();
            OneHotPoly::<AkitaField, u8>::new(k, indices).unwrap()
        })
        .collect::<Vec<_>>();
    let trace_sources = columns.iter().collect::<Vec<_>>();
    let materialized_sources = materialized_columns.iter().collect::<Vec<_>>();
    let num_blocks = RootPolyShape::<AkitaField, D>::num_ring_elems(source).div_ceil(num_positions);
    let backend = test_backend();
    let challenges = (0..num_blocks * COLUMNS)
        .map(|block| SparseChallenge {
            positions: vec![0, (block % (D - 1) + 1) as u32].into(),
            coeffs: vec![1, -1].into(),
        })
        .collect::<Vec<_>>();
    for num_chunks in [1, 2, 4, 8] {
        let challenge_set =
            Challenges::from_sparse(challenges.clone(), num_blocks, COLUMNS).unwrap();
        let ranges = akita_params::dyadic_block_ranges(num_blocks, num_chunks).unwrap();
        if num_chunks > num_blocks {
            assert!(ranges.iter().any(Range::is_empty));
        }
        let plan = if num_chunks == 1 {
            DecomposeFoldBatchPlan::Sparse {
                challenges: &challenges,
                num_positions_per_block: num_positions,
                num_digits,
                log_basis: 3,
            }
        } else {
            DecomposeFoldBatchPlan::SparseChunked {
                challenges: &challenge_set,
                chunk_ranges: &ranges,
                num_positions_per_block: num_positions,
                num_digits,
                log_basis: 3,
            }
        };
        let streamed = <TestBackend as OpeningBatchKernel<
            TraceOneHotColumnBatchView<'_, D>,
            AkitaField,
            D,
        >>::decompose_fold_batch(
            &backend,
            None,
            <TraceOneHotColumn as RootOpeningSource<AkitaField, D>>::opening_batch(&trace_sources)
                .unwrap(),
            plan,
        )
        .unwrap();
        let materialized = <TestBackend as OpeningBatchKernel<
            OneHotBatchView<'_, AkitaField, D, u8>,
            AkitaField,
            D,
        >>::decompose_fold_batch(
            &backend,
            None,
            <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_batch(
                &materialized_sources,
            )
            .unwrap(),
            plan,
        )
        .unwrap();
        assert_eq!(streamed, materialized);
        for mode in [
            DecomposeRotationMode::Dense,
            DecomposeRotationMode::Sparse,
            DecomposeRotationMode::Compact,
        ] {
            let chunks = decompose_fold_columns_with_mode::<D>(
                source,
                &challenges,
                &ranges,
                num_positions,
                num_digits,
                mode,
            )
            .unwrap();
            let response = if num_chunks == 1 {
                CpuFoldResponses::sparse(chunks.into_iter().next().unwrap())
            } else {
                CpuFoldResponses::chunked::<D>(chunks).unwrap()
            };
            assert_eq!(
                response, materialized,
                "D={D}, chunks={num_chunks}, mode={mode:?}"
            );
        }
    }
    let prepared = packing_point::<D>(
        source.num_vars,
        RootPolyShape::<AkitaField, D>::num_ring_elems(source),
        num_positions,
    );
    let plan = SubringCoefficientPackingPlan { point: &prepared };
    let streamed = <TestBackend as SubringCoefficientPackingBatchKernel<
        TraceOneHotColumnBatchView<'_, D>,
        AkitaField,
        AkitaField,
        D,
    >>::coefficient_packing_partials_batch(
        &backend,
        None,
        <TraceOneHotColumn as RootOpeningSource<AkitaField, D>>::opening_batch(&trace_sources)
            .unwrap(),
        plan,
    )
    .unwrap();
    let materialized = <TestBackend as SubringCoefficientPackingBatchKernel<
        OneHotBatchView<'_, AkitaField, D, u8>,
        AkitaField,
        AkitaField,
        D,
    >>::coefficient_packing_partials_batch(
        &backend,
        None,
        <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_batch(
            &materialized_sources,
        )
        .unwrap(),
        plan,
    )
    .unwrap();
    assert_eq!(streamed, materialized);
}

#[test]
fn blockwise_production_kernels_match_materialized_onehot() {
    assert_production_kernels_match_materialized::<64>(256, 32, 16, None, 2);
    assert_production_kernels_match_materialized::<64>(256, 32, 1, None, 2);
    assert_production_kernels_match_materialized::<128>(256, 32, 16, None, 2);
    assert_production_kernels_match_materialized::<256>(256, 32, 16, None, 2);
    assert_production_kernels_match_materialized::<512>(256, 32, 8, None, 2);
    assert_production_kernels_match_materialized::<64>(16, 32, 4, None, 2);
    assert_production_kernels_match_materialized::<128>(16, 32, 2, None, 2);
    assert_production_kernels_match_materialized::<256>(16, 32, 2, None, 2);
    assert_production_kernels_match_materialized::<512>(16, 32, 1, None, 2);
    assert_production_kernels_match_materialized::<64>(16, 32, 16, None, 2);
    assert_production_kernels_match_materialized::<64>(16, 32, 32, Some(1), 2);
    assert_production_kernels_match_materialized::<64>(256, 32, 16, Some(0), 2);
    assert_production_kernels_match_materialized::<128>(16, 32, 8, None, 2);
    assert_production_kernels_match_materialized::<256>(16, 32, 4, None, 2);
    assert_production_kernels_match_materialized::<512>(16, 32, 2, None, 2);
    assert_production_kernels_match_materialized::<64>(256, 32, 16, Some(1), 2);
    assert_production_kernels_match_materialized::<64>(16, 32, 4, Some(1), 2);
    assert_production_kernels_match_materialized::<64>(256, 32, 16, Some(1), 1);
    assert_production_kernels_match_materialized::<64>(16, 32, 4, Some(1), 1);
    assert_production_kernels_match_materialized::<256>(256, 32, 16, Some(1), 1);
}

fn batch_decompose_test_source<const D: usize>() -> Vec<TraceOneHotColumn> {
    TraceOneHotColumn::new(
        16,
        D,
        Arc::new(TestRows {
            rows: 32,
            columns: 3,
            k: 16,
            committed_zero_column: None,
        }),
    )
    .unwrap()
}

fn batch_decompose_error<const D: usize>(
    source: &[TraceOneHotColumn],
    plan: DecomposeFoldBatchPlan<'_>,
) -> Option<AkitaError> {
    let sources = source.iter().collect::<Vec<_>>();
    let backend = test_backend();
    <TestBackend as OpeningBatchKernel<
        TraceOneHotColumnBatchView<'_, D>,
        AkitaField,
        D,
    >>::decompose_fold_batch(
        &backend,
        None,
        <TraceOneHotColumn as RootOpeningSource<AkitaField, D>>::opening_batch(&sources).unwrap(),
        plan,
    )
    .err()
}

fn sparse_challenges(count: usize) -> Vec<SparseChallenge> {
    vec![
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        count
    ]
}

#[test]
fn batch_decompose_rejects_zero_positions_per_block() {
    const D: usize = 64;
    let source = batch_decompose_test_source::<D>();
    let error = batch_decompose_error::<D>(
        &source,
        DecomposeFoldBatchPlan::Sparse {
            challenges: &[],
            num_positions_per_block: 0,
            num_digits: 2,
            log_basis: 3,
        },
    )
    .unwrap();
    assert!(matches!(error, AkitaError::InvalidInput(_)));
}

#[test]
fn batch_decompose_rejects_malformed_challenge_count() {
    const D: usize = 64;
    const POSITIONS_PER_BLOCK: usize = 2;
    let source = batch_decompose_test_source::<D>();
    let num_blocks = RootPolyShape::<AkitaField, D>::num_live_ring_elems(&source[0])
        .div_ceil(POSITIONS_PER_BLOCK);
    let challenges = Challenges::from_sparse(
        sparse_challenges(2 * num_blocks * source.len()),
        num_blocks,
        2 * source.len(),
    )
    .unwrap();
    let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, 2).unwrap();
    let error = batch_decompose_error::<D>(
        &source,
        DecomposeFoldBatchPlan::SparseChunked {
            challenges: &challenges,
            chunk_ranges: &chunk_ranges,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits: 2,
            log_basis: 3,
        },
    )
    .unwrap();
    assert!(matches!(
        error,
        AkitaError::InvalidSize { expected, actual }
            if expected == num_blocks * source.len() && actual == 2 * num_blocks * source.len()
    ));
}

#[test]
fn batch_decompose_rejects_nonuniform_live_block_geometry() {
    const D: usize = 64;
    const POSITIONS_PER_BLOCK: usize = 2;
    let source = batch_decompose_test_source::<D>();
    let num_blocks = RootPolyShape::<AkitaField, D>::num_live_ring_elems(&source[0])
        .div_ceil(POSITIONS_PER_BLOCK);
    let declared_num_blocks = num_blocks + 1;
    let challenges = Challenges::from_sparse(
        sparse_challenges(declared_num_blocks * source.len()),
        declared_num_blocks,
        source.len(),
    )
    .unwrap();
    let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, 2).unwrap();
    let error = batch_decompose_error::<D>(
        &source,
        DecomposeFoldBatchPlan::SparseChunked {
            challenges: &challenges,
            chunk_ranges: &chunk_ranges,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits: 2,
            log_basis: 3,
        },
    )
    .unwrap();
    assert!(matches!(
        error,
        AkitaError::InvalidInput(message)
            if message == "batched decompose_fold sources have different live-block extents"
    ));
}

#[test]
fn batch_decompose_rejects_noncanonical_chunk_ranges() {
    const D: usize = 64;
    const POSITIONS_PER_BLOCK: usize = 2;
    let source = batch_decompose_test_source::<D>();
    let num_blocks = RootPolyShape::<AkitaField, D>::num_live_ring_elems(&source[0])
        .div_ceil(POSITIONS_PER_BLOCK);
    let challenges = Challenges::from_sparse(
        sparse_challenges(num_blocks * source.len()),
        num_blocks,
        source.len(),
    )
    .unwrap();
    let chunk_ranges = [0..num_blocks, 0..0];
    let error = batch_decompose_error::<D>(
        &source,
        DecomposeFoldBatchPlan::SparseChunked {
            challenges: &challenges,
            chunk_ranges: &chunk_ranges,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits: 2,
            log_basis: 3,
        },
    )
    .unwrap();
    assert!(matches!(
        error,
        AkitaError::InvalidInput(message) if message == "noncanonical fold chunk ranges"
    ));
}

#[test]
fn small_k256_blocks_commit_like_materialized_onehot() {
    const D: usize = 64;
    const K: usize = 256;
    // The native commitment comparison uses the admitted (20, 29) fixture.
    const ROWS: usize = 1 << 12;
    const COLUMNS: usize = 29;
    const POSITIONS_PER_BLOCK: usize = 2;
    let columns = TraceOneHotColumn::new(
        K,
        D,
        Arc::new(TestRows {
            rows: ROWS,
            columns: COLUMNS,
            k: K,
            committed_zero_column: Some(0),
        }),
    )
    .unwrap();
    let source = &columns[0];
    let materialized_columns = (0..COLUMNS)
        .map(|column| {
            OneHotPoly::<AkitaField, u8>::new(
                K,
                (0..ROWS)
                    .map(|row| {
                        let hot = ((row * (2 * column + 1) + column) % K) as u8;
                        (hot != 0 || column == 0).then_some(hot)
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        1,
        1,
        SetupMatrixCapacity {
            num_field_elements: D * POSITIONS_PER_BLOCK,
        },
    )
    .unwrap();
    let plan = CommitInnerPlan {
        ring_dimension: D,
        num_live_blocks: RootPolyShape::<AkitaField, D>::num_ring_elems(&source)
            / POSITIONS_PER_BLOCK,
        n_a: 1,
        num_positions_per_block: POSITIONS_PER_BLOCK,
        num_digits_inner: 1,
        log_basis_inner: 1,
    };

    let streamed = commit_columns::<D>(&setup.expanded, source, plan).unwrap();

    // Oracle: Akita's canonical one-hot table, one ring per D coefficients,
    // under the single-digit inner map rows[b] = sum_p A[0][p] * ring(b * P + p).
    let a_view = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(plan.n_a, POSITIONS_PER_BLOCK)
        .unwrap();
    let a_wide = a_view
        .rows()
        .next()
        .unwrap()
        .iter()
        .map(AkitaWideRing::<D>::from_ring)
        .collect::<Vec<_>>();
    for (streamed, materialized_source) in streamed.iter().zip(&materialized_columns) {
        let coefficients = materialized_source.source_coefficients().unwrap();
        let mut expected = vec![AkitaWideRing::<D>::zero(); plan.num_live_blocks];
        for (ring, ring_coefficients) in coefficients.chunks_exact(D).enumerate() {
            for (index, coefficient) in ring_coefficients.iter().enumerate() {
                if *coefficient == AkitaField::one() {
                    a_wide[ring % POSITIONS_PER_BLOCK]
                        .shift_accumulate_into(&mut expected[ring / POSITIONS_PER_BLOCK], index);
                } else {
                    assert_eq!(*coefficient, AkitaField::from_u64(0));
                }
            }
        }
        let expected = expected
            .into_iter()
            .map(|value| value.reduce::<AkitaField>())
            .collect::<Vec<_>>();
        assert_eq!(streamed.as_ring_slice::<D>().unwrap(), expected.as_slice());
    }
    let (pcs_setup, _) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        source.num_vars,
        COLUMNS,
        [9; 32],
        K,
        AkitaScheduleArtifacts::shared_from_default_directory(),
    ))
    .unwrap();
    let native = materialized_columns
        .iter()
        .map(|poly| OneHotPolynomial::new(K, poly.indices().to_vec()))
        .collect();
    let (native_commitment, _) =
        AkitaScheme::commit_one_hot_group_owned(&pcs_setup, [9; 32], native).unwrap();
    let (streamed_commitment, _) =
        AkitaScheme::commit_trace_one_hot(&pcs_setup, [9; 32], Arc::clone(&source.rows), &[])
            .unwrap();
    assert_eq!(streamed_commitment, native_commitment);
}

#[derive(Debug)]
struct InvalidSelectorRows;

impl TraceOneHotRows for InvalidSelectorRows {
    fn num_rows(&self) -> usize {
        32
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn fill_row(&self, _row: usize, selected_rows: &mut [u8]) {
        selected_rows[0] = 16;
    }

    fn committed_digit_zero_mask(&self, _row: usize) -> u64 {
        0
    }
}

#[test]
fn coefficient_packing_rejects_invalid_selector() {
    const D: usize = 64;
    let columns = TraceOneHotColumn::new(16, D, Arc::new(InvalidSelectorRows)).unwrap();
    let source = &columns[0];
    let num_live_positions = RootPolyShape::<AkitaField, D>::num_ring_elems(&source);
    let prepared = packing_point::<D>(source.num_vars, num_live_positions, 4);
    let error = coefficient_packing_partials_columns::<AkitaField, D>(
        source,
        SubringCoefficientPackingPlan { point: &prepared },
    )
    .expect_err("selector outside K must reject");
    assert!(error.to_string().contains("outside K=16"));
}

#[path = "tests/trace_reads.rs"]
mod trace_reads;
