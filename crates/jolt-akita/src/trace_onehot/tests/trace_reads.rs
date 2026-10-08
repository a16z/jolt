use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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
fn fused_kernels_read_each_trace_row_once() {
    const D: usize = 64;
    const ROWS: usize = 32;
    const NUM_POSITIONS: usize = 4;
    let fills = Arc::new(AtomicUsize::new(0));
    let columns = TraceOneHotColumn::new(
        16,
        D,
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
    let source = &columns[0];
    let num_blocks =
        RootPolyShape::<AkitaField, D>::num_ring_elems(&source).div_ceil(NUM_POSITIONS);
    let challenges = (0..num_blocks * columns.len())
        .map(|block| SparseChallenge {
            positions: vec![0, (block % (D - 1) + 1) as u32].into(),
            coeffs: vec![1, -1].into(),
        })
        .collect::<Vec<_>>();
    let sources = columns.iter().collect::<Vec<_>>();
    let backend = test_backend();
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        1,
        1,
        SetupMatrixCapacity {
            num_field_elements: D * NUM_POSITIONS,
        },
    )
    .unwrap();
    let _ = commit_columns::<D>(
        &setup.expanded,
        source,
        CommitInnerPlan {
            ring_dimension: D,
            num_live_blocks: num_blocks,
            n_a: 1,
            num_positions_per_block: NUM_POSITIONS,
            num_digits_inner: 1,
            log_basis_inner: 1,
        },
    )
    .unwrap();
    assert_eq!(fills.swap(0, Ordering::Relaxed), ROWS);
    let challenge_set = Challenges::from_sparse(challenges, num_blocks, columns.len()).unwrap();
    let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, 2).unwrap();
    let chunks = <TestBackend as OpeningBatchKernel<
        TraceOneHotColumnBatchView<'_, D>,
        AkitaField,
        D,
    >>::decompose_fold_batch(
        &backend,
        None,
        <TraceOneHotColumn as RootOpeningSource<AkitaField, D>>::opening_batch(&sources).unwrap(),
        DecomposeFoldBatchPlan::SparseChunked {
            challenges: &challenge_set,
            chunk_ranges: &chunk_ranges,
            num_positions_per_block: NUM_POSITIONS,
            num_digits: 2,
            log_basis: 3,
        },
    )
    .unwrap();
    assert_eq!(chunks.chunk_count(), 2);
    assert_eq!(fills.load(Ordering::Relaxed), ROWS);
}

#[test]
fn coefficient_packing_reads_each_trace_row_once() {
    const D: usize = 64;
    const ROWS: usize = 32;
    let fills = Arc::new(AtomicUsize::new(0));
    let columns = TraceOneHotColumn::new(
        16,
        D,
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
    let source = &columns[0];
    let num_live_positions = RootPolyShape::<AkitaField, D>::num_ring_elems(&source);
    let prepared = packing_point::<D>(source.num_vars, num_live_positions, 4);
    let _ = coefficient_packing_partials_columns::<AkitaField, D>(
        source,
        SubringCoefficientPackingPlan { point: &prepared },
    )
    .unwrap();
    assert_eq!(fills.load(Ordering::Relaxed), ROWS);
}
