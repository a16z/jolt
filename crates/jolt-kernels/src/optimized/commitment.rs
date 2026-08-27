//! The optimized witness-commitment kernel: the reference consumer's exact
//! per-column call sequences, parallelized.
//!
//! The reference kernel streams `row_width`-cycle chunks and advances every
//! column's commitment state serially per chunk — each tier-1 group operation
//! (one MSM or batch addition per column per chunk) runs alone on the calling
//! thread, so the whole commit is wall-clock serial. This kernel restores the
//! legacy prover's parallel shape on the same call sequences:
//!
//! - The stream delivers *superchunks* (many `row_width` windows at once), so
//!   one extraction pass fans out to a `(column × window)` rayon grid via the
//!   [`StreamingCommitment`] batch entry points (`feed_i128_rows`,
//!   `process_one_hot_chunks`).
//! - Tier-2 finishes (one multi-pairing per column) run in parallel across
//!   columns.
//!
//! Per column the fed windows, their order, and the finish calls are exactly
//! the reference kernel's, so commitments and hints are byte-identical.
//! The materializing modes (address-major order, widened grids) and advice
//! commits delegate to the reference kernel unchanged.

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_witness::{
    stream_witnesses, JoltWitnessOracle, RandomAccessRows, RowSource, StreamConsumer, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::commitment::{
    finish_streamed_one_hot_prepared, finish_streamed_prepared, CommitWitness, CommitmentGrid,
    CommittedColumnsWitness, ModeStreamingCommitment, WitnessCommitment,
};
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::{KernelError, OptimizedBackend, ProofSession, ReferenceBackend};

/// Superchunk ceiling — the measured 64-thread optimum.
#[cfg(feature = "parallel")]
const SUPERCHUNK_CYCLES_MAX: usize = 1 << 21;

/// Cycles per superchunk, scaled to the pool. The extracted bundle is 80
/// bytes per cycle and the pipeline retains two buffers, so applying the
/// 64-thread optimum to every host needlessly reserves about 320 MiB.
fn superchunk_cycles() -> usize {
    #[cfg(feature = "parallel")]
    {
        (rayon::current_num_threads() << 15)
            .next_power_of_two()
            .clamp(1 << 17, SUPERCHUNK_CYCLES_MAX)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1 << 17
    }
}

#[cfg(feature = "parallel")]
const COLLECT_PAR_CHUNK: usize = 1 << 12;

impl<F, PCS> CommitWitness<F, PCS> for OptimizedBackend
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    // The backend-neutral `commit_witness` span lives at the stage-0 call
    // site. A per-implementation span with the same label would double its
    // aggregated duration in profiling summaries.
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        source: &dyn RowSource,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>> {
        let cycles = 1usize << grid.log_t;
        let row_width = grid.num_columns();

        if grid.order != TracePolynomialOrder::CycleMajor || row_width > cycles {
            // Materializing modes are off the streaming hot path; the
            // reference kernel's one-table-per-column passes serve them.
            return ReferenceBackend.commit_witness(session, source, ids, grid, setup);
        }

        commit_streaming(source, ids, grid, setup, superchunk_cycles())
    }

    fn commit_advice(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>> {
        // Advice grids are small single-column commits; the reference pass
        // is already the right shape.
        ReferenceBackend.commit_advice(session, witness, id, grid, setup)
    }
}

/// The streaming commit pass at an explicit superchunk width (tests shrink
/// it to force multi-delivery sequencing; production uses
/// [`superchunk_cycles`]).
fn commit_streaming<F, PCS>(
    source: &dyn RowSource,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
    superchunk_cycles: usize,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    let cycles = 1usize << grid.log_t;
    let row_width = grid.num_columns();
    // Superchunk width: a power-of-two window count (both factors are powers
    // of two), so every delivery is whole windows.
    let windows = (superchunk_cycles / row_width).clamp(1, cycles / row_width);
    let superchunk = row_width * windows;
    // Slice-backed sources pipeline the extraction of the next superchunk
    // against the commit grid of the current one; re-emulating sources
    // alternate the two phases through the sequential walk.
    #[cfg(feature = "parallel")]
    if let Some(access) = source.random_access() {
        if cycles <= access.cycles() {
            return commit_pipelined(&access, ids, grid, setup, superchunk);
        }
    }
    commit_streamed(source, ids, grid, setup, superchunk)
}

/// The chunk-walk commit pass: extraction and the commit grid alternate, a
/// barrier between every phase.
fn commit_streamed<F, PCS>(
    source: &dyn RowSource,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
    superchunk: usize,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    let cycles = 1usize << grid.log_t;
    let row_width = grid.num_columns();
    let kinds = column_kinds(ids, grid)?;
    let mut consumers = (BatchedColumns::<F, PCS>::begin(
        &kinds, row_width, grid, setup,
    ),);
    stream_witnesses(source, 0..cycles, superchunk, &mut consumers)?;
    Ok(package::<F, PCS>(consumers.0.finish(setup), ids))
}

/// The pipelined commit pass over a slice-backed source: while the column
/// grid advances over superchunk `k`, workers extract superchunk `k + 1`
/// into the spare buffer (two reused buffers, swapped per delivery). Per
/// column the fed windows, their order, and the finish calls are exactly
/// the chunk walk's — the pipeline only overlaps extraction with group
/// arithmetic, so commitments and hints are byte-identical.
#[cfg(feature = "parallel")]
fn collect_range_into(
    access: &RandomAccessRows,
    range: std::ops::Range<usize>,
    out: &mut Vec<CommittedColumnsWitness>,
) -> Result<(), WitnessError> {
    out.clear();
    let start = range.start;
    let count = range.end - start;
    out.reserve(count);
    let spare = &mut out.spare_capacity_mut()[..count];
    spare
        .par_chunks_mut(COLLECT_PAR_CHUNK)
        .enumerate()
        .try_for_each(|(chunk, destination)| {
            let base = start + chunk * COLLECT_PAR_CHUNK;
            for (offset, slot) in destination.iter_mut().enumerate() {
                let _ = slot.write(access.window(base + offset)?);
            }
            Ok::<_, WitnessError>(())
        })?;
    // SAFETY: every slot was initialized above; the row type is `Copy`.
    unsafe { out.set_len(count) };
    Ok(())
}

#[cfg(feature = "parallel")]
fn commit_pipelined<F, PCS>(
    access: &RandomAccessRows,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
    superchunk: usize,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    let cycles = 1usize << grid.log_t;
    let row_width = grid.num_columns();
    let kinds = column_kinds(ids, grid)?;
    let mut state = BatchedColumns::<F, PCS>::begin(&kinds, row_width, grid, setup);

    let mut front: Vec<CommittedColumnsWitness> = Vec::new();
    let mut back: Vec<CommittedColumnsWitness> = Vec::new();
    let mut end = superchunk.min(cycles);
    collect_range_into(access, 0..end, &mut front)?;
    loop {
        let next_end = (end + superchunk).min(cycles);
        let (fill, ()) = rayon::join(
            || {
                if end < next_end {
                    collect_range_into(access, end..next_end, &mut back).map(|()| true)
                } else {
                    Ok(false)
                }
            },
            || state.consume(&front),
        );
        if !fill? {
            break;
        }
        core::mem::swap(&mut front, &mut back);
        end = next_end;
    }
    Ok(package::<F, PCS>(state.finish(setup), ids))
}

/// Zips finished per-column outputs back to their polynomial ids.
fn package<F, PCS>(
    outputs: Vec<(PCS::Output, PCS::OpeningHint)>,
    ids: &[JoltCommittedPolynomial],
) -> Vec<WitnessCommitment<PCS>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    outputs
        .into_iter()
        .zip(ids)
        .map(|((commitment, hint), &id)| WitnessCommitment {
            id,
            commitment,
            hint,
        })
        .collect()
}

/// One column's in-progress commitment — the reference kernel's states,
/// advanced a superchunk at a time through the batch entry points.
enum ColumnCommitState<PCS: ModeStreamingCommitment> {
    Increment {
        kind: ColumnKind,
        partial: PCS::PartialCommitment,
    },
    OneHot {
        kind: ColumnKind,
        context: PCS::OneHotStreamContext,
        chunk_commitments: Vec<PCS::OneHotChunkCommitment>,
    },
}

/// The superchunked commit consumer: every column advances over the same
/// window sequence as the reference kernel, columns in parallel and windows
/// in parallel inside each batch call.
struct BatchedColumns<'a, F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment>
{
    columns: Vec<ColumnCommitState<PCS>>,
    one_hot_k: usize,
    row_width: usize,
    /// Row windows delivered so far — the increment columns' tier-2 row
    /// count (one-hot columns aggregate `one_hot_k` rows per window).
    windows_fed: usize,
    setup: &'a PCS::ProverSetup,
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment>
    BatchedColumns<'a, F, PCS>
{
    fn begin(
        kinds: &[ColumnKind],
        row_width: usize,
        grid: CommitmentGrid,
        setup: &'a PCS::ProverSetup,
    ) -> Self {
        let columns = kinds
            .iter()
            .map(|&kind| {
                if kind.is_one_hot() {
                    ColumnCommitState::OneHot {
                        kind,
                        context: PCS::begin_one_hot_column_major_stream(setup, row_width),
                        chunk_commitments: Vec::new(),
                    }
                } else {
                    ColumnCommitState::Increment {
                        kind,
                        partial: PCS::begin(setup),
                    }
                }
            })
            .collect();
        Self {
            columns,
            one_hot_k: 1usize << grid.log_k_chunk,
            row_width,
            windows_fed: 0,
            setup,
        }
    }

    fn finish(self, setup: &PCS::ProverSetup) -> Vec<(PCS::Output, PCS::OpeningHint)> {
        let one_hot_k = self.one_hot_k;
        // Every column's tier-2 pairs against the same setup generator
        // prefix; prepare it once for the whole pass. One-hot columns
        // aggregate `windows · one_hot_k` rows, increment columns `windows`.
        let max_rows = self
            .columns
            .iter()
            .map(|column| match column {
                ColumnCommitState::Increment { .. } => self.windows_fed,
                ColumnCommitState::OneHot { .. } => self.windows_fed * one_hot_k,
            })
            .max()
            .unwrap_or(0);
        let prep = PCS::prepare_tier2(setup, max_rows);
        let finish_column = |column: ColumnCommitState<PCS>| match column {
            ColumnCommitState::Increment { partial, .. } => {
                finish_streamed_prepared::<PCS>(partial, setup, &prep)
            }
            ColumnCommitState::OneHot {
                chunk_commitments, ..
            } => {
                finish_streamed_one_hot_prepared::<PCS>(setup, one_hot_k, &chunk_commitments, &prep)
            }
        };
        #[cfg(feature = "parallel")]
        {
            self.columns.into_par_iter().map(finish_column).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.columns.into_iter().map(finish_column).collect()
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment> StreamConsumer
    for BatchedColumns<'_, F, PCS>
{
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        debug_assert!(
            chunk.len().is_multiple_of(self.row_width),
            "superchunk must be whole rows"
        );
        self.windows_fed += chunk.len() / self.row_width;
        let row_width = self.row_width;
        let one_hot_k = self.one_hot_k;
        let setup = self.setup;
        // Columns feed by closure straight off the shared bundle chunk —
        // the commit windows materialize their own values worker-side, so
        // no per-column batch staging exists at any superchunk size.
        let advance = |column: &mut ColumnCommitState<PCS>| match column {
            ColumnCommitState::Increment { kind, partial } => {
                PCS::feed_i128_rows_with(
                    partial,
                    |index| kind.increment(&chunk[index]),
                    chunk.len(),
                    row_width,
                    setup,
                );
            }
            ColumnCommitState::OneHot {
                kind,
                context,
                chunk_commitments,
            } => {
                chunk_commitments.extend(PCS::process_one_hot_chunks_with(
                    context,
                    setup,
                    one_hot_k,
                    |index| kind.hot_address(&chunk[index]),
                    chunk.len(),
                    row_width,
                ));
            }
        };
        #[cfg(feature = "parallel")]
        self.columns.par_iter_mut().for_each(advance);
        #[cfg(not(feature = "parallel"))]
        self.columns.iter_mut().for_each(advance);
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    #![expect(clippy::unwrap_used, reason = "test module")]

    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_witness::RowSource;

    use super::{commit_streamed, commit_streaming};
    use crate::commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
    use crate::optimized::testing::{with_ram_fixture, FixtureShape, RamOp};
    use crate::{OptimizedBackend, ProofSession, ReferenceBackend};

    fn assert_same_commitments(
        reference: &[WitnessCommitment<DoryScheme>],
        optimized: &[WitnessCommitment<DoryScheme>],
    ) {
        assert_eq!(reference.len(), optimized.len());
        for (reference, optimized) in reference.iter().zip(optimized) {
            assert_eq!(reference.id, optimized.id);
            assert_eq!(
                reference.commitment, optimized.commitment,
                "{:?} commitment diverged",
                reference.id
            );
            assert_eq!(
                reference.hint, optimized.hint,
                "{:?} hint diverged",
                reference.id
            );
        }
    }

    /// The optimized streaming pass must reproduce the reference kernel's
    /// commitments and hints exactly, both when a superchunk covers the whole
    /// trace (one multi-window delivery) and when it is forced down to one
    /// window (multi-delivery sequencing).
    #[test]
    fn optimized_commit_matches_reference() {
        let shape = FixtureShape {
            log_t: 6,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 2, post: 17 },
            RamOp::Read { word: 2 },
            RamOp::None,
            RamOp::Write { word: 5, post: 3 },
            RamOp::Read { word: 5 },
            RamOp::Write { word: 2, post: 9 },
            RamOp::Read { word: 3 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let ids: Vec<JoltCommittedPolynomial> = witness
                .committed_order()
                .unwrap()
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
            let grid = CommitmentGrid {
                total_vars: 4 + shape.log_t,
                log_t: shape.log_t,
                log_k_chunk: 4,
                order: TracePolynomialOrder::CycleMajor,
            };
            assert!(
                grid.num_columns() < 1usize << shape.log_t,
                "fixture must exercise the streaming path with multiple windows"
            );
            let setup = DoryScheme::setup_prover(grid.total_vars);
            let source: &dyn RowSource = witness;

            let reference = <ReferenceBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
                &ReferenceBackend,
                &mut ProofSession::default(),
                source,
                &ids,
                grid,
                &setup,
            )
            .unwrap();

            let optimized = <OptimizedBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
                &OptimizedBackend,
                &mut ProofSession::default(),
                source,
                &ids,
                grid,
                &setup,
            )
            .unwrap();
            assert_same_commitments(&reference, &optimized);

            let single_window_superchunks =
                commit_streaming::<Fr, DoryScheme>(source, &ids, grid, &setup, grid.num_columns())
                    .unwrap();
            assert_same_commitments(&reference, &single_window_superchunks);

            // Both delivery shapes pinned explicitly: the chunk-walk pass
            // (re-emulating sources) and the pipelined pass (slice-backed
            // sources), at whole-trace and single-window superchunks.
            let streamed =
                commit_streamed::<Fr, DoryScheme>(source, &ids, grid, &setup, grid.num_columns())
                    .unwrap();
            assert_same_commitments(&reference, &streamed);
            #[cfg(feature = "parallel")]
            {
                let access = source.random_access().unwrap();
                let pipelined = super::commit_pipelined::<Fr, DoryScheme>(
                    &access,
                    &ids,
                    grid,
                    &setup,
                    grid.num_columns(),
                )
                .unwrap();
                assert_same_commitments(&reference, &pipelined);
            }
        });
    }
}
