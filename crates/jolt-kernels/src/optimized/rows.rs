//! Access shapes over a slice-backed witness source, derived from the
//! plane's one escape hatch ([`jolt_witness::RowSource::shared_rows`]): the
//! borrowed random-access view and the index-parallel collectors the
//! optimized kernels use in place of the sequential chunk walk. Extraction
//! is pure per cycle window, so collected values are identical to the
//! walk's — the padding and lookahead semantics are pinned against
//! [`jolt_witness::collect_bundles`] by the parity tests below.

#[cfg(feature = "parallel")]
use core::ops::Range;

use jolt_program::execution::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
#[cfg(feature = "parallel")]
use jolt_witness::{par_collect_windows, FirstErrorLatch};
use jolt_witness::{SharedTraceRows, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A slice-backed source's rows and extraction context, for index-parallel
/// collection. Cycles at or beyond the physical rows are padding (default)
/// rows, exactly as the sequential walk serves them.
pub(crate) struct RandomAccessRows<'a> {
    /// The physical trace rows.
    rows: &'a [TraceRow],
    /// The padded cycle-domain size (`2^log_t`); the lookahead window ends
    /// here regardless of the collected range.
    cycles: usize,
    /// The extraction environment shared by every window.
    env: WitnessEnv<'a>,
    /// The padding row served at and beyond the physical trace, shared by
    /// every window (and every thread) of the view.
    padding: TraceRow,
}

impl RandomAccessRows<'_> {
    /// Extracts the bundle at `index` with the sequential walk's one-row
    /// lookahead window (padding rows at and beyond the physical trace, no
    /// lookahead at the end of the cycle domain). Pure per index — callers
    /// extract from any thread in any order.
    #[inline]
    pub(crate) fn window<B: WitnessBundle>(&self, index: usize) -> Result<B, WitnessError> {
        let current = self.rows.get(index).unwrap_or(&self.padding);
        let next =
            (index + 1 < self.cycles).then(|| self.rows.get(index + 1).unwrap_or(&self.padding));
        B::from_row(current, next, &self.env)
    }
}

/// The borrowed random-access view over a shared-rows handle.
pub(crate) trait SharedRowsExt {
    fn view(&self) -> RandomAccessRows<'_>;
}

impl SharedRowsExt for SharedTraceRows {
    fn view(&self) -> RandomAccessRows<'_> {
        RandomAccessRows {
            rows: &self.rows,
            cycles: self.cycles,
            env: self.env(),
            padding: TraceRow::default(),
        }
    }
}

/// The parallel scatter grain of the range collector: big enough to
/// amortize rayon dispatch, small enough to load-balance skewed extraction.
#[cfg(feature = "parallel")]
const COLLECT_PAR_CHUNK: usize = 1 << 12;

/// Index-parallel bundle collection over a random-access view, mapped
/// through `pack` element by element (so packed forms never stage the wide
/// bundle): values are identical to the sequential pass's — extraction is
/// pure per cycle window, and the lookahead row at the end of the range
/// matches the chunk walk's `next_after`. `V: Copy` makes the collectors'
/// leak-free-on-error invariant compiler-checked.
pub(crate) fn collect_par_map<B: WitnessBundle, V: Copy + Send>(
    access: &RandomAccessRows<'_>,
    cycles: usize,
    pack: impl Fn(B) -> V + Send + Sync,
) -> Result<Vec<V>, WitnessError> {
    let window = |index: usize| access.window::<B>(index).map(&pack);
    #[cfg(feature = "parallel")]
    return par_collect_windows(cycles, window);
    #[cfg(not(feature = "parallel"))]
    (0..cycles).map(window).collect()
}

/// [`collect_par_map`] without the packing step: index-parallel collection
/// of the bundles themselves.
pub(crate) fn collect_bundles_par<B: WitnessBundle + Copy + Send>(
    access: &RandomAccessRows<'_>,
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    collect_par_map(access, cycles, |bundle: B| bundle)
}

/// Index-parallel collection of one cycle sub-range into a reusable buffer
/// (cleared first, allocation kept): the pipelining collector's shape — a
/// caller overlapping extraction with downstream work re-fills two buffers
/// alternately instead of allocating per delivery. `B: Copy` rules out drop
/// obligations, so leaving the buffer cleared on error leaks nothing; the
/// lowest-index error wins (deterministic across runs).
#[cfg(feature = "parallel")]
pub(crate) fn collect_range_into<B: WitnessBundle + Copy + Send>(
    access: &RandomAccessRows<'_>,
    range: Range<usize>,
    out: &mut Vec<B>,
) -> Result<(), WitnessError> {
    out.clear();
    {
        let start = range.start;
        let count = range.end.saturating_sub(start);
        out.reserve(count);
        let spare = &mut out.spare_capacity_mut()[..count];
        let error = FirstErrorLatch::new();
        spare
            .par_chunks_mut(COLLECT_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_index, destination)| {
                let base = start + chunk_index * COLLECT_PAR_CHUNK;
                for (offset, slot) in destination.iter_mut().enumerate() {
                    match access.window::<B>(base + offset) {
                        Ok(bundle) => {
                            let _ = slot.write(bundle);
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return;
                        }
                    }
                }
            });
        if let Some(failure) = error.take() {
            return Err(failure);
        }
        // SAFETY: the error latch is empty, so every chunk ran to completion
        // and initialized all `count` slots of the spare capacity above.
        unsafe { out.set_len(count) };
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test-only module")]
mod tests {
    use super::*;
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::{NextUnexpandedPc, UnexpandedPc};
    use jolt_witness::{collect_bundles, RowSource, WitnessBundle};

    /// A two-column window bundle: `NextUnexpandedPc` reads the lookahead
    /// row, so padding and end-of-domain semantics are both exercised.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
    struct WindowBundle {
        pc: UnexpandedPc,
        next_pc: NextUnexpandedPc,
    }

    /// The view's window extraction must reproduce the sequential walk's
    /// values exactly — including padding rows and the end-of-domain
    /// lookahead cutoff.
    #[test]
    fn view_collection_matches_the_chunked_walk() {
        with_sample_backend(|backend| {
            let shared = backend.shared_rows().unwrap();
            let sequential: Vec<WindowBundle> = collect_bundles(backend, shared.cycles).unwrap();
            let view = shared.view();
            let parallel: Vec<WindowBundle> = collect_bundles_par(&view, shared.cycles).unwrap();
            assert_eq!(sequential, parallel);

            #[cfg(feature = "parallel")]
            {
                let mut ranged: Vec<WindowBundle> = Vec::new();
                let split = shared.cycles / 2;
                collect_range_into(&view, 0..split, &mut ranged).unwrap();
                assert_eq!(sequential[..split], ranged);
                collect_range_into(&view, split..shared.cycles, &mut ranged).unwrap();
                assert_eq!(sequential[split..], ranged);
            }
        });
    }
}
