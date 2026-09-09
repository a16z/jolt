#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests assert valid kernel geometry and explicit rejection"
)]

use super::*;
use std::sync::Arc;

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_prover::backend::OneHotBatchView;
use akita_prover::compute::{
    DecomposeFoldPlan, OpeningFoldKernel, OpeningFoldPlan, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPlan,
};
use akita_prover::{CpuBackend, OneHotPoly, RootOpeningSource, RootPolyMeta, RootPolyShape};
use akita_types::{
    BasisMode, PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
};
use jolt_field::{One, Ring};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::source::{TracePackedOneHotBatchView, TracePackedOneHotView};
use crate::AkitaField;

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
    let source = TracePackedOneHot::new(
        k,
        64,
        8,
        Arc::new(TestRows {
            rows,
            columns: 3,
            k,
            committed_zero_column,
        }),
    )
    .unwrap();
    let segment_rings = source.segment_ring_elems::<D>().unwrap();
    let mut actual = Vec::new();
    let view =
        <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap();
    visit_segment_ring_range::<D>(view.source(), 0, segment_rings, |ring, contributions| {
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
    assert_ring_mapping::<64>(256, 32, Some(1));
}

fn assert_k16_shift_groups<const D: usize>() {
    const COLUMNS: usize = 5;
    let rows_per_ring = D / 16;
    let mut selected_rows = vec![NO_SELECTED_ROW; rows_per_ring * COLUMNS];
    let committed_zero_masks = vec![0u64; rows_per_ring];
    for row in 0..rows_per_ring {
        let shared_hot = ((row + 1) % 15 + 1) as u8;
        selected_rows[row * COLUMNS] = shared_hot;
        selected_rows[row * COLUMNS + 1] = shared_hot;
        selected_rows[row * COLUMNS + 2] = shared_hot;
        selected_rows[row * COLUMNS + 3] = ((2 * row + 3) % 15 + 1) as u8;
        selected_rows[row * COLUMNS + 4] = if row == 1 {
            NO_SELECTED_ROW
        } else {
            ((3 * row + 5) % 15 + 1) as u8
        };
    }

    let source: CyclotomicRing<AkitaField, D> =
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            AkitaField::from_u64((index + 1) as u64)
        }));
    let source: AkitaWideRing<D> = AkitaWideRing::from_ring(&source);
    let mut actual = vec![AkitaWideRing::zero(); COLUMNS];
    for (chunk, chunk_rows) in selected_rows.chunks_exact(4 * COLUMNS).enumerate() {
        let masks = &committed_zero_masks[4 * chunk..4 * chunk + 4];
        let mut groups = K16FourRowShiftGroups::new(COLUMNS, 4 * chunk).unwrap();
        assert!(groups.build(chunk_rows, masks, COLUMNS));
        groups.accumulate(&source, &mut actual, 0, 1, chunk_rows, masks, COLUMNS);
    }

    let mut expected = vec![AkitaWideRing::zero(); COLUMNS];
    for (row, row_indices) in selected_rows.chunks_exact(COLUMNS).enumerate() {
        for (column, &hot) in row_indices.iter().enumerate() {
            if hot != NO_SELECTED_ROW {
                source.shift_accumulate_into(&mut expected[column], 16 * row + usize::from(hot));
            }
        }
    }
    let actual = actual
        .into_iter()
        .map(|value| value.reduce::<AkitaField>())
        .collect::<Vec<_>>();
    let expected = expected
        .into_iter()
        .map(|value| value.reduce::<AkitaField>())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn k16_shared_shift_groups_cover_adaptive_dimensions() {
    assert_k16_shift_groups::<64>();
    assert_k16_shift_groups::<128>();
    assert_k16_shift_groups::<256>();
}

#[test]
fn constructor_enforces_selector_capacity() {
    let rows = Arc::new(TestRows {
        rows: 32,
        columns: 9,
        k: 16,
        committed_zero_column: None,
    });
    assert!(TracePackedOneHot::new(16, 64, 8, rows).is_err());
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
    assert_deferred_fp128_shift_accumulator::<512>();
}

fn assert_opening_kernels_match_materialized<const D: usize>(
    k: usize,
    rows: usize,
    num_positions: usize,
    committed_zero_column: Option<usize>,
) {
    const COLUMNS: usize = 3;
    const CAPACITY: usize = 8;
    let source = TracePackedOneHot::new(
        k,
        64,
        CAPACITY,
        Arc::new(TestRows {
            rows,
            columns: COLUMNS,
            k,
            committed_zero_column,
        }),
    )
    .unwrap();
    let packed_indices = (0..CAPACITY)
        .flat_map(|column| {
            (0..rows).map(move |row| {
                let selected_row = ((row * (2 * column + 1) + column) % k) as u8;
                (column < COLUMNS && (selected_row != 0 || committed_zero_column == Some(column)))
                    .then_some(selected_row)
            })
        })
        .collect();
    let materialized_source = OneHotPoly::<AkitaField, u8>::new(k, packed_indices).unwrap();
    let num_blocks = <TracePackedOneHot as RootPolyShape<AkitaField, D>>::num_ring_elems(&source)
        / num_positions;
    let live_weights = (0..num_blocks)
        .map(|index| AkitaField::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let position_weights = (0..num_positions)
        .map(|index| AkitaField::from_u64((3 * index + 1) as u64))
        .collect::<Vec<_>>();
    let fold_plan = OpeningFoldPlan::Base {
        live_block_weights: &live_weights,
        position_weights: &position_weights,
        num_positions_per_block: num_positions,
    };
    let backend = CpuBackend::DEFAULT;
    let streamed = <CpuBackend as OpeningFoldKernel<
            TracePackedOneHotView<'_, D>,
            AkitaField,
            D,
        >>::evaluate_and_fold(
            &backend,
            None,
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap(),
            fold_plan,
        )
        .unwrap();
    let materialized = <CpuBackend as OpeningFoldKernel<_, AkitaField, D>>::evaluate_and_fold(
        &backend,
        None,
        <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_view(
            &materialized_source,
        )
        .unwrap(),
        fold_plan,
    )
    .unwrap();
    assert_eq!(streamed, materialized);

    let challenges = (0..num_blocks)
        .map(|block| SparseChallenge {
            positions: vec![0, (block % (D - 1) + 1) as u32].into(),
            coeffs: vec![1, -1].into(),
        })
        .collect::<Vec<_>>();
    let decompose_plan = DecomposeFoldPlan {
        challenges: &challenges,
        num_positions_per_block: num_positions,
        num_digits: 2,
        log_basis: 3,
    };
    let streamed = <CpuBackend as OpeningFoldKernel<
            TracePackedOneHotView<'_, D>,
            AkitaField,
            D,
        >>::decompose_fold(
            &backend,
            None,
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap(),
            decompose_plan,
        )
        .unwrap();
    let materialized = <CpuBackend as OpeningFoldKernel<_, AkitaField, D>>::decompose_fold(
        &backend,
        None,
        <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_view(
            &materialized_source,
        )
        .unwrap(),
        decompose_plan,
    )
    .unwrap();
    assert_eq!(streamed, materialized);
    let view =
        <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap();
    let source = view.source();
    let dense = decompose_fold_packed_with_mode::<D>(
        source,
        &challenges,
        num_positions,
        2,
        DecomposeRotationMode::Dense,
    )
    .unwrap();
    let sparse = decompose_fold_packed_with_mode::<D>(
        source,
        &challenges,
        num_positions,
        2,
        DecomposeRotationMode::Sparse,
    )
    .unwrap();
    let compact = decompose_fold_packed_with_mode::<D>(
        source,
        &challenges,
        num_positions,
        2,
        DecomposeRotationMode::Compact,
    )
    .unwrap();
    assert_eq!(dense, materialized);
    assert_eq!(sparse, materialized);
    assert_eq!(compact, materialized);

    let source_num_vars = RootPolyMeta::<AkitaField>::num_vars(&source);
    let num_live_positions = RootPolyShape::<AkitaField, D>::num_ring_elems(&source);
    let prepared_point = packing_point::<D>(source_num_vars, num_live_positions, num_positions);
    let packing_plan = SubringCoefficientPackingPlan {
        point: &prepared_point,
    };
    let trace_sources = [source];
    let trace_view =
        <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_batch(&trace_sources)
            .unwrap();
    let streamed =
        <CpuBackend as SubringCoefficientPackingBatchKernel<
            TracePackedOneHotBatchView<'_, D>,
            AkitaField,
            AkitaField,
            D,
        >>::coefficient_packing_partials_batch(&backend, None, trace_view, packing_plan)
        .unwrap();
    let materialized_sources = [&materialized_source];
    let materialized_view =
        <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_batch(
            &materialized_sources,
        )
        .unwrap();
    let materialized = <CpuBackend as SubringCoefficientPackingBatchKernel<
        OneHotBatchView<'_, AkitaField, D, u8>,
        AkitaField,
        AkitaField,
        D,
    >>::coefficient_packing_partials_batch(
        &backend, None, materialized_view, packing_plan
    )
    .unwrap();
    assert_eq!(streamed, materialized);
}

#[test]
fn d128_auto_uses_compact_rotations() {
    let challenges = [SparseChallenge {
        positions: vec![0, 127].into(),
        coeffs: vec![1, -1].into(),
    }];
    let rotations =
        prepare_rotations::<128>(&challenges, None, 1, DecomposeRotationMode::Auto).unwrap();
    assert!(matches!(rotations, PreparedRotations::Compact(_)));
}

#[test]
fn blockwise_opening_kernels_match_materialized_onehot() {
    assert_opening_kernels_match_materialized::<64>(256, 32, 16, None);
    assert_opening_kernels_match_materialized::<128>(256, 32, 16, None);
    assert_opening_kernels_match_materialized::<256>(256, 32, 16, None);
    assert_opening_kernels_match_materialized::<512>(256, 32, 8, None);
    assert_opening_kernels_match_materialized::<64>(16, 32, 4, None);
    assert_opening_kernels_match_materialized::<128>(16, 32, 2, None);
    assert_opening_kernels_match_materialized::<256>(16, 32, 2, None);
    assert_opening_kernels_match_materialized::<512>(16, 32, 1, None);
    assert_opening_kernels_match_materialized::<64>(16, 32, 16, None);
    assert_opening_kernels_match_materialized::<128>(16, 32, 8, None);
    assert_opening_kernels_match_materialized::<256>(16, 32, 4, None);
    assert_opening_kernels_match_materialized::<512>(16, 32, 2, None);
    assert_opening_kernels_match_materialized::<64>(256, 32, 16, Some(1));
    assert_opening_kernels_match_materialized::<64>(16, 32, 4, Some(1));
}

#[derive(Debug)]
struct CountingRows {
    inner: TestRows,
    fills: Arc<AtomicUsize>,
}

impl TraceOneHotRows for CountingRows {
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        let _ = self.fills.fetch_add(1, Ordering::Relaxed);
        self.inner.fill_row(row, selected_rows);
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        self.inner.committed_digit_zero_mask(row)
    }
}

#[test]
fn coefficient_packing_reads_each_trace_row_once() {
    const D: usize = 64;
    const ROWS: usize = 32;
    let fills = Arc::new(AtomicUsize::new(0));
    let source = TracePackedOneHot::new(
        16,
        D,
        8,
        Arc::new(CountingRows {
            inner: TestRows {
                rows: ROWS,
                columns: 3,
                k: 16,
                committed_zero_column: None,
            },
            fills: Arc::clone(&fills),
        }),
    )
    .unwrap();
    let num_live_positions = RootPolyShape::<AkitaField, D>::num_ring_elems(&source);
    let prepared = packing_point::<D>(source.num_vars, num_live_positions, 4);
    let _ = coefficient_packing_partials_packed::<AkitaField, D>(
        &source,
        SubringCoefficientPackingPlan { point: &prepared },
    )
    .unwrap();
    assert_eq!(fills.load(Ordering::Relaxed), ROWS);
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
    let source = TracePackedOneHot::new(16, D, 1, Arc::new(InvalidSelectorRows)).unwrap();
    let num_live_positions = RootPolyShape::<AkitaField, D>::num_ring_elems(&source);
    let prepared = packing_point::<D>(source.num_vars, num_live_positions, 4);
    let error = coefficient_packing_partials_packed::<AkitaField, D>(
        &source,
        SubringCoefficientPackingPlan { point: &prepared },
    )
    .expect_err("selector outside K must reject");
    assert!(error.to_string().contains("outside K=16"));
}

/// The production D=512 / K=256 commit path (two trace rows per ring, deferred
/// limb accumulation over K256 row tiles) must equal a canonical negacyclic
/// shift-accumulate over the same setup rows, including the committed-zero
/// mask semantics and more than one flush window.
#[test]
fn d512_k256_commit_matches_canonical_accumulate() {
    use super::traversal::row_is_committed;
    use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, RootCommitSource};
    use akita_types::SetupMatrixCapacity;

    const D: usize = 512;
    const K: usize = 256;
    const COLUMNS: usize = 11;
    const CAPACITY: usize = 16;
    const ROWS: usize = 1 << 15;
    const POSITIONS: usize = 4096;

    let rows = TestRows {
        rows: ROWS,
        columns: COLUMNS,
        k: K,
        committed_zero_column: Some(3),
    };
    let source = TracePackedOneHot::new(
        K,
        D,
        CAPACITY,
        Arc::new(TestRows {
            rows: ROWS,
            columns: COLUMNS,
            k: K,
            committed_zero_column: Some(3),
        }),
    )
    .unwrap();
    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: POSITIONS,
        num_digits_inner: 1,
        log_basis_inner: 8,
    };
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        RootPolyMeta::<AkitaField>::num_vars(&source),
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * POSITIONS * D,
        },
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let prepared = cpu.prepare_setup(&setup).unwrap();
    let witness = cpu
        .commit_inner_group(
            &prepared,
            vec![
                <TracePackedOneHot as RootCommitSource<AkitaField, D>>::commit_view(&source)
                    .unwrap(),
            ],
            plan,
        )
        .unwrap();
    let output = witness[0].inner_rows.as_ring_slice::<D>().unwrap();

    let a_view = cpu
        .prepared_expanded_setup(&prepared)
        .shared_matrix()
        .ring_view::<D>(plan.n_a, POSITIONS)
        .unwrap();
    let a_row = a_view.rows().next().unwrap();
    let rows_per_ring = D / K;
    let blocks_per_column = ROWS / rows_per_ring / POSITIONS;
    let mut expected = vec![CyclotomicRing::<AkitaField, D>::zero(); CAPACITY * blocks_per_column];
    let mut selected = vec![NO_SELECTED_ROW; COLUMNS];
    for row in 0..ROWS {
        rows.fill_row(row, &mut selected);
        let mask = rows.committed_digit_zero_mask(row);
        let ring = row / rows_per_ring;
        let row_offset = row % rows_per_ring;
        let (block, position) = (ring / POSITIONS, ring % POSITIONS);
        for (column, &hot) in selected.iter().enumerate() {
            if row_is_committed(hot, mask, column) {
                a_row[position].shift_accumulate_into(
                    &mut expected[column * blocks_per_column + block],
                    row_offset * K + usize::from(hot),
                );
            }
        }
    }
    assert_eq!(output.len(), expected.len());
    assert!(output.iter().zip(&expected).all(|(got, want)| got == want));
}
