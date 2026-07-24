//! The reference witness-commitment kernel: a consumer of the witness
//! stream.
//!
//! Under the cycle-major order every committed column is fed from ONE fused
//! pass over the trace: the commit consumer implements [`StreamConsumer`]
//! over the [`CommittedColumnsWitness`] fact bundle, holds every column's
//! partial commitment state (the runtime arity lives here, in the consumer),
//! and per row window feeds dense columns through the
//! [`StreamingCommitment::feed`] family and one-hot columns through the
//! column-major one-hot path — the same per-column call sequences as
//! committing each polynomial separately, so the commitments are identical.
//!
//! The materializing modes run one pass per column to keep peak memory at
//! one grid table: under the address-major order coefficients scatter
//! cycle-block-strided across the whole grid (`index = t · cycle_stride +
//! k · one_hot_stride`, dense polynomials at address slot zero) and a
//! widened grid (committed-program candidates) packs multiple `k`-blocks of
//! the flat `(K × T)` one-hot matrix per committed row — both feed the
//! materialized table's rows, the same per-row MSMs legacy's materialized
//! commits run, full matrix height included (its trailing identity rows are
//! part of the wire hint).

use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, TracePolynomialOrder,
};
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::{stream_witnesses, JoltWitnessOracle, RowSource, StreamConsumer};

use crate::commitment::{
    CommitWitness, CommitmentGrid, CommittedColumnsWitness, WitnessCommitment,
};
use crate::{KernelError, ProofSession, ReferenceBackend};

impl<F, PCS> CommitWitness<F, PCS> for ReferenceBackend
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        source: &dyn RowSource,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>> {
        let kinds = column_kinds(ids, grid)?;
        let cycles = 1usize << grid.log_t;
        let row_width = grid.num_columns();

        if grid.order == TracePolynomialOrder::CycleMajor && row_width <= cycles {
            // The streaming-friendly mode: one fused pass feeds every column.
            let mut consumers = (FusedColumns::<F, PCS>::begin(
                &kinds, row_width, grid, setup,
            ),);
            stream_witnesses(source, 0..cycles, row_width, &mut consumers)?;
            return Ok(consumers
                .0
                .finish(setup)
                .into_iter()
                .zip(ids)
                .map(|((commitment, hint), &id)| WitnessCommitment {
                    id,
                    commitment,
                    hint,
                })
                .collect());
        }

        // Materializing modes: one pass and one grid table per column.
        kinds
            .into_iter()
            .zip(ids)
            .map(|(kind, &id)| {
                let mut consumers = (MaterializedColumn::<F>::begin(kind, grid),);
                stream_witnesses(source, 0..cycles, row_width, &mut consumers)?;
                let table = consumers.0.table;
                let mut partial = PCS::begin(setup);
                for row in table.chunks(row_width) {
                    PCS::feed(&mut partial, row, setup);
                }
                let (commitment, hint) = PCS::finish_with_hint(partial, setup);
                Ok(WitnessCommitment {
                    id,
                    commitment,
                    hint,
                })
            })
            .collect()
    }

    fn commit_advice(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>> {
        // Advice grids are cycle-major with no one-hot placement, and the
        // column is small: materialize it and feed dense rows.
        let values = witness.oracle_table(JoltPolynomialId::Committed(id))?;
        let mut partial = PCS::begin(setup);
        for row in values.chunks(grid.num_columns()) {
            PCS::feed(&mut partial, row, setup);
        }
        let (commitment, hint) = PCS::finish_with_hint(partial, setup);
        Ok(WitnessCommitment {
            id,
            commitment,
            hint,
        })
    }
}

/// A committed column's derivation from the fact bundle: the increments
/// directly, the one-hots through the consumer-held chunk selector.
#[derive(Clone, Copy, Debug)]
enum ColumnKind {
    RdInc,
    RamInc,
    InstructionRa(RaChunkSelector),
    BytecodeRa(RaChunkSelector),
    RamRa(RaChunkSelector),
}

impl ColumnKind {
    const fn is_one_hot(self) -> bool {
        matches!(
            self,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_)
        )
    }

    fn increment(self, row: &CommittedColumnsWitness) -> i128 {
        match self {
            Self::RdInc => row.rd_inc.0,
            Self::RamInc => row.ram_inc.0,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => {
                unreachable!("one-hot columns go through hot_address")
            }
        }
    }

    fn hot_address(self, row: &CommittedColumnsWitness) -> Option<usize> {
        match self {
            Self::InstructionRa(selector) => Some(selector.chunk_u128(row.lookup_index.0)),
            Self::BytecodeRa(selector) => row.bytecode_pc.0.map(|pc| selector.chunk_usize(pc)),
            Self::RamRa(selector) => row
                .ram_address
                .0
                .map(|address| selector.chunk_usize(address as usize)),
            Self::RdInc | Self::RamInc => unreachable!("increments go through increment"),
        }
    }
}

/// Resolve `ids` to column derivations. Family sizes come from the ids
/// themselves (the committed order carries whole families); the chunk width
/// is the grid's.
fn column_kinds<F: Field>(
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
) -> Result<Vec<ColumnKind>, KernelError<F>> {
    let family_size = |matches: fn(JoltCommittedPolynomial) -> bool| {
        ids.iter().copied().filter(|&id| matches(id)).count()
    };
    let instruction_chunks =
        family_size(|id| matches!(id, JoltCommittedPolynomial::InstructionRa(_)));
    let bytecode_chunks = family_size(|id| matches!(id, JoltCommittedPolynomial::BytecodeRa(_)));
    let ram_chunks = family_size(|id| matches!(id, JoltCommittedPolynomial::RamRa(_)));
    let selector = |index: usize, chunks: usize| {
        RaChunkSelector::new(index, chunks, grid.log_k_chunk).map_err(KernelError::from)
    };
    ids.iter()
        .map(|&id| match id {
            JoltCommittedPolynomial::RdInc => Ok(ColumnKind::RdInc),
            JoltCommittedPolynomial::RamInc => Ok(ColumnKind::RamInc),
            JoltCommittedPolynomial::InstructionRa(index) => Ok(ColumnKind::InstructionRa(
                selector(index, instruction_chunks)?,
            )),
            JoltCommittedPolynomial::BytecodeRa(index) => {
                Ok(ColumnKind::BytecodeRa(selector(index, bytecode_chunks)?))
            }
            JoltCommittedPolynomial::RamRa(index) => {
                Ok(ColumnKind::RamRa(selector(index, ram_chunks)?))
            }
            _ => Err(KernelError::InvalidGeometry {
                reason: format!(
                    "{id:?} is not a trace-derived column (advice commits through commit_advice)"
                ),
            }),
        })
        .collect()
}

/// The fused cycle-major commit consumer: every column's in-progress
/// commitment, advanced per row window.
struct FusedColumns<'a, F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment> {
    columns: Vec<ColumnCommitState<PCS>>,
    one_hot_k: usize,
    setup: &'a PCS::ProverSetup,
    /// Scratch buffers for one row window's column values, reused across
    /// windows and columns to avoid per-chunk allocation.
    increments: Vec<i128>,
    hot_addresses: Vec<Option<usize>>,
}

/// One column's in-progress commitment: dense columns accumulate a partial
/// commitment through the `feed` family; one-hot columns accumulate
/// per-window chunk commitments through the column-major one-hot stream.
enum ColumnCommitState<PCS: StreamingCommitment> {
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

impl<'a, F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment>
    FusedColumns<'a, F, PCS>
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
            setup,
            increments: Vec::with_capacity(row_width),
            hot_addresses: Vec::with_capacity(row_width),
        }
    }

    fn finish(self, setup: &PCS::ProverSetup) -> Vec<(PCS::Output, PCS::OpeningHint)> {
        let one_hot_k = self.one_hot_k;
        self.columns
            .into_iter()
            .map(|column| match column {
                ColumnCommitState::Increment { partial, .. } => {
                    PCS::finish_with_hint(partial, setup)
                }
                ColumnCommitState::OneHot {
                    chunk_commitments, ..
                } => PCS::finish_one_hot_column_major_chunks(setup, one_hot_k, &chunk_commitments),
            })
            .collect()
    }
}

impl<F: Field, PCS: CommitmentScheme<Field = F> + StreamingCommitment> StreamConsumer
    for FusedColumns<'_, F, PCS>
{
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        for column in &mut self.columns {
            match column {
                ColumnCommitState::Increment { kind, partial } => {
                    self.increments.clear();
                    self.increments
                        .extend(chunk.iter().map(|row| kind.increment(row)));
                    PCS::feed_i128(partial, &self.increments, self.setup);
                }
                ColumnCommitState::OneHot {
                    kind,
                    context,
                    chunk_commitments,
                } => {
                    self.hot_addresses.clear();
                    self.hot_addresses
                        .extend(chunk.iter().map(|row| kind.hot_address(row)));
                    chunk_commitments.push(PCS::process_one_hot_chunk(
                        context,
                        self.setup,
                        self.one_hot_k,
                        &self.hot_addresses,
                    ));
                }
            }
        }
    }
}

/// A materializing per-column consumer: scatters one column into its full
/// grid table (address-major strides, or the flat `(K × T)` layout on
/// widened cycle-major grids), fed row-by-row afterwards.
struct MaterializedColumn<F> {
    kind: ColumnKind,
    table: Vec<F>,
    cycle: usize,
    cycle_stride: usize,
    one_hot_stride: usize,
    flat_cycles: Option<usize>,
}

impl<F: Field> MaterializedColumn<F> {
    fn begin(kind: ColumnKind, grid: CommitmentGrid) -> Self {
        // Widened cycle-major grids materialize one-hots as the flat (K × T)
        // matrix and dense columns in the plain cycle-major layout;
        // address-major grids materialize the full strided table.
        let (table_len, flat_cycles) = if grid.order == TracePolynomialOrder::CycleMajor {
            if kind.is_one_hot() {
                (
                    (1usize << grid.log_k_chunk) << grid.log_t,
                    Some(1usize << grid.log_t),
                )
            } else {
                (1usize << grid.log_t, None)
            }
        } else {
            (1usize << grid.total_vars, None)
        };
        Self {
            kind,
            table: vec![F::zero(); table_len],
            cycle: 0,
            cycle_stride: if grid.order == TracePolynomialOrder::AddressMajor {
                grid.cycle_stride()
            } else {
                1
            },
            one_hot_stride: if grid.order == TracePolynomialOrder::AddressMajor {
                grid.one_hot_stride()
            } else {
                0
            },
            flat_cycles,
        }
    }
}

impl<F: Field> StreamConsumer for MaterializedColumn<F> {
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        for row in chunk {
            if self.kind.is_one_hot() {
                if let Some(k) = self.kind.hot_address(row) {
                    // Selector masks bound k below the grid's chunk width.
                    let index = match self.flat_cycles {
                        Some(cycles) => k * cycles + self.cycle,
                        None => self.cycle * self.cycle_stride + k * self.one_hot_stride,
                    };
                    self.table[index] = F::one();
                }
            } else {
                self.table[self.cycle * self.cycle_stride] = F::from_i128(self.kind.increment(row));
            }
            self.cycle += 1;
        }
    }
}
