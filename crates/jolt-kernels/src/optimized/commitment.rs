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
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::{
    collect_range_into, stream_witnesses, JoltWitnessOracle, RowSource, StreamConsumer,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::commitment::{
    CommitWitness, CommitmentGrid, CommittedColumnsWitness, WitnessCommitment,
};
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::{KernelError, OptimizedBackend, ProofSession, ReferenceBackend};

/// Cycles per superchunk. The dominant stage-0 wall cost on a many-core
/// host is the per-superchunk join (its critical path is one window's
/// serial MSM), so fewer, larger superchunks win: 2^17 → 2^19 measured
/// 59.3s → 43.1s whole-prove at 2^25 on 64 threads. The extracted bundle
/// buffer (64 B/cycle) and per-column scratch stay tens of megabytes.
const SUPERCHUNK_CYCLES: usize = 1 << 19;

impl<F, PCS> CommitWitness<F, PCS> for OptimizedBackend
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    #[tracing::instrument(
        skip_all,
        name = "commit_witness",
        fields(columns = ids.len(), total_vars = grid.total_vars)
    )]
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

        commit_streaming(source, ids, grid, setup, SUPERCHUNK_CYCLES)
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
/// [`SUPERCHUNK_CYCLES`]).
fn commit_streaming<F, PCS>(
    source: &dyn RowSource,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
    superchunk_cycles: usize,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
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
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
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
fn commit_pipelined<F, PCS>(
    access: &jolt_witness::RandomAccessRows<'_>,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
    superchunk: usize,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
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
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
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
enum ColumnCommitState<PCS: StreamingCommitment> {
    Increment {
        kind: ColumnKind,
        partial: PCS::PartialCommitment,
        /// Reused per-delivery staging for the column's increments —
        /// allocated once, so 30 columns × hundreds of deliveries don't
        /// churn the allocator (arena high-water marks follow transients).
        scratch: Vec<i128>,
    },
    OneHot {
        kind: ColumnKind,
        context: PCS::OneHotStreamContext,
        chunk_commitments: Vec<PCS::OneHotChunkCommitment>,
        /// Reused per-delivery staging for the column's hot addresses.
        scratch: Vec<Option<usize>>,
    },
}

/// The superchunked commit consumer: every column advances over the same
/// window sequence as the reference kernel, columns in parallel and windows
/// in parallel inside each batch call.
struct BatchedColumns<'a, F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment> {
    columns: Vec<ColumnCommitState<PCS>>,
    one_hot_k: usize,
    row_width: usize,
    setup: &'a PCS::ProverSetup,
}

impl<'a, F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment>
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
                        scratch: Vec::new(),
                    }
                } else {
                    ColumnCommitState::Increment {
                        kind,
                        partial: PCS::begin(setup),
                        scratch: Vec::new(),
                    }
                }
            })
            .collect();
        Self {
            columns,
            one_hot_k: 1usize << grid.log_k_chunk,
            row_width,
            setup,
        }
    }

    fn finish(self, setup: &PCS::ProverSetup) -> Vec<(PCS::Output, PCS::OpeningHint)> {
        let one_hot_k = self.one_hot_k;
        let finish_column = |column: ColumnCommitState<PCS>| match column {
            ColumnCommitState::Increment { partial, .. } => PCS::finish_with_hint(partial, setup),
            ColumnCommitState::OneHot {
                chunk_commitments, ..
            } => PCS::finish_one_hot_column_major_chunks(setup, one_hot_k, &chunk_commitments),
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

impl<F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment> StreamConsumer
    for BatchedColumns<'_, F, PCS>
{
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        debug_assert!(
            chunk.len().is_multiple_of(self.row_width),
            "superchunk must be whole rows"
        );
        let row_width = self.row_width;
        let one_hot_k = self.one_hot_k;
        let setup = self.setup;
        let advance = |column: &mut ColumnCommitState<PCS>| match column {
            ColumnCommitState::Increment {
                kind,
                partial,
                scratch,
            } => {
                scratch.clear();
                scratch.extend(chunk.iter().map(|row| kind.increment(row)));
                PCS::feed_i128_rows(partial, scratch, row_width, setup);
            }
            ColumnCommitState::OneHot {
                kind,
                context,
                chunk_commitments,
                scratch,
            } => {
                scratch.clear();
                scratch.extend(chunk.iter().map(|row| kind.hot_address(row)));
                chunk_commitments.extend(PCS::process_one_hot_chunks(
                    context, setup, one_hot_k, scratch, row_width,
                ));
            }
        };
        #[cfg(feature = "parallel")]
        self.columns.par_iter_mut().for_each(advance);
        #[cfg(not(feature = "parallel"))]
        self.columns.iter_mut().for_each(advance);
    }
}

#[cfg(test)]
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
