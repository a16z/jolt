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

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
};
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, TracePolynomialOrder,
};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::witnesses::RaChunkSelector;
#[cfg(feature = "field-inline")]
use jolt_witness::JoltWitnessPlane;
use jolt_witness::{stream_witnesses, JoltWitnessOracle, RowSource, StreamConsumer};

#[cfg(feature = "field-inline")]
use crate::commitment::FieldInlineWitnessCommitment;
use crate::commitment::{
    finish_streamed, finish_streamed_one_hot, CommitWitness, CommitmentGrid,
    CommittedColumnsWitness, ModeStreamingCommitment, WitnessCommitment,
};
use crate::{KernelError, ProofSession, ReferenceBackend};

impl<F, PCS> CommitWitness<F, PCS> for ReferenceBackend
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    // The backend-neutral `commit_witness` span lives at the stage-0 call
    // boundary (`crates/jolt-prover/src/stages/stage0.rs`), so every
    // `CommitWitness` implementation inherits it — see the taxonomy's
    // kernel-seam contract.
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
                let (commitment, hint) = finish_streamed::<PCS>(partial, setup);
                Ok(WitnessCommitment {
                    id,
                    commitment,
                    hint,
                })
            })
            .collect()
    }

    // Instrumented at the stage-0 call boundary, like `commit_witness`.
    #[cfg(feature = "field-inline")]
    fn commit_field_inline_witness(
        &self,
        _session: &mut ProofSession,
        source: &dyn JoltWitnessPlane<F>,
        ids: &[FieldInlineCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<FieldInlineWitnessCommitment<PCS>>, KernelError<F>> {
        commit_field_inline_columns::<F, PCS>(source, ids, grid, setup)
    }

    // Instrumented at the stage-0 call boundary, like `commit_witness`.
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
        let (commitment, hint) = finish_streamed::<PCS>(partial, setup);
        Ok(WitnessCommitment {
            id,
            commitment,
            hint,
        })
    }
}

/// A committed column's derivation from the fact bundle: the increments
/// directly, the one-hots through the consumer-held chunk selector. Shared
/// with the optimized joint-opening kernel — the opened values must be the
/// committed values, so both derive through this one type.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ColumnKind {
    RdInc,
    RamInc,
    InstructionRa(RaChunkSelector),
    BytecodeRa(RaChunkSelector),
    RamRa(RaChunkSelector),
    /// Dense over the trace domain like the increments, but field-valued:
    /// committed from the plane's field-inline oracle, never from the
    /// [`CommittedColumnsWitness`] stream.
    #[cfg(feature = "field-inline")]
    FieldRdInc,
}

impl ColumnKind {
    pub(crate) const fn is_one_hot(self) -> bool {
        matches!(
            self,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_)
        )
    }

    pub(crate) fn increment(self, row: &CommittedColumnsWitness) -> i128 {
        match self {
            Self::RdInc => row.rd_inc.0,
            Self::RamInc => row.ram_inc.0,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => {
                unreachable!("one-hot columns go through hot_address")
            }
            #[cfg(feature = "field-inline")]
            Self::FieldRdInc => {
                unreachable!("field-inline columns commit from the field-inline oracle")
            }
        }
    }

    pub(crate) fn hot_address(self, row: &CommittedColumnsWitness) -> Option<usize> {
        match self {
            Self::InstructionRa(selector) => Some(selector.chunk_u128(row.lookup_index.0)),
            Self::BytecodeRa(selector) => Some(selector.chunk_usize(row.bytecode_pc.0)),
            Self::RamRa(selector) => row
                .ram_address
                .0
                .map(|address| selector.chunk_usize(address as usize)),
            Self::RdInc | Self::RamInc => unreachable!("increments go through increment"),
            #[cfg(feature = "field-inline")]
            Self::FieldRdInc => {
                unreachable!("field-inline columns commit from the field-inline oracle")
            }
        }
    }
}

/// A committed-column id at the commit-kernel seam: the jolt family always,
/// the field-inline family under the composed protocol. Kernel-local
/// composite — the jolt-claims id namespaces stay disjoint (the same pattern
/// as jolt-verifier's `VerifierOpeningId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommittedColumnId {
    Jolt(JoltCommittedPolynomial),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineCommittedPolynomial),
}

impl From<JoltCommittedPolynomial> for CommittedColumnId {
    fn from(id: JoltCommittedPolynomial) -> Self {
        Self::Jolt(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineCommittedPolynomial> for CommittedColumnId {
    fn from(id: FieldInlineCommittedPolynomial) -> Self {
        Self::FieldInline(id)
    }
}

/// Resolve `ids` to column derivations. Family sizes come from the ids
/// themselves (the committed order carries whole families); the chunk width
/// is the grid's. Generic over the id family so the jolt call sites stay
/// unchanged while the field-inline pass resolves through the same table.
pub(crate) fn column_kinds<F: JoltField, Id: Copy + Into<CommittedColumnId>>(
    ids: &[Id],
    grid: CommitmentGrid,
) -> Result<Vec<ColumnKind>, KernelError<F>> {
    let ids: Vec<CommittedColumnId> = ids.iter().map(|&id| id.into()).collect();
    let family_size = |matches: fn(JoltCommittedPolynomial) -> bool| {
        ids.iter()
            .filter(|&&id| match id {
                CommittedColumnId::Jolt(id) => matches(id),
                #[cfg(feature = "field-inline")]
                CommittedColumnId::FieldInline(_) => false,
            })
            .count()
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
            CommittedColumnId::Jolt(id) => match id {
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
            },
            #[cfg(feature = "field-inline")]
            CommittedColumnId::FieldInline(FieldInlineCommittedPolynomial::FieldRdInc) => {
                Ok(ColumnKind::FieldRdInc)
            }
        })
        .collect()
}

/// The shared field-inline commit pass, used by every `CommitWitness` tier:
/// each FR column is dense over the trace domain and placed exactly like the
/// jolt increment columns (contiguous cycle-major; address slot zero of each
/// cycle block address-major), so the stage-8 embedding treats `FieldRdInc`
/// like `RdInc`.
#[cfg(feature = "field-inline")]
pub(crate) fn commit_field_inline_columns<F, PCS>(
    source: &dyn JoltWitnessPlane<F>,
    ids: &[FieldInlineCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
) -> Result<Vec<FieldInlineWitnessCommitment<PCS>>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment,
{
    let kinds = column_kinds::<F, _>(ids, grid)?;
    let oracle = source.field_inline().ok_or(KernelError::Unsupported {
        reason: "field-inline commit requires a witness plane serving the field-inline oracle",
    })?;
    let cycles = 1usize << grid.log_t;
    ids.iter()
        .zip(kinds)
        .map(|(&id, kind)| {
            if kind.is_one_hot() {
                return Err(KernelError::InvalidGeometry {
                    reason: format!("{id:?} is not a dense trace-domain column"),
                });
            }
            let values = oracle.oracle_table(FieldInlinePolynomialId::Committed(id))?;
            if values.len() != cycles {
                return Err(KernelError::InvalidGeometry {
                    reason: format!(
                        "{id:?} has {} evaluations, the trace domain holds {cycles}",
                        values.len()
                    ),
                });
            }
            let mut partial = PCS::begin(setup);
            let width = grid.num_columns();
            match grid.order {
                TracePolynomialOrder::CycleMajor => {
                    for row in values.chunks(width) {
                        PCS::feed(&mut partial, row, setup);
                    }
                }
                // Address-major: cycle `t` sits at grid index `t · stride`,
                // everything else is zero. Stream the grid row by row without
                // materializing the K·T table — the rows holding no cycle
                // slot go through `feed_zeros`.
                TracePolynomialOrder::AddressMajor => {
                    let stride = grid.cycle_stride();
                    let rows = (1usize << grid.total_vars) / width;
                    let mut row: Vec<F> = vec![F::zero(); width];
                    let mut zero_rows = 0usize;
                    for row_index in 0..rows {
                        let start = row_index * width;
                        let first_cycle = start.div_ceil(stride).min(cycles);
                        let end_cycle = ((start + width - 1) / stride + 1).min(cycles);
                        if first_cycle >= end_cycle {
                            zero_rows += 1;
                            continue;
                        }
                        PCS::feed_zeros(&mut partial, width, zero_rows, setup);
                        zero_rows = 0;
                        for cycle in first_cycle..end_cycle {
                            row[cycle * stride - start] = values[cycle];
                        }
                        PCS::feed(&mut partial, &row, setup);
                        for cycle in first_cycle..end_cycle {
                            row[cycle * stride - start] = F::zero();
                        }
                    }
                    PCS::feed_zeros(&mut partial, width, zero_rows, setup);
                }
            }
            let (commitment, hint) = finish_streamed::<PCS>(partial, setup);
            Ok(FieldInlineWitnessCommitment {
                id,
                commitment,
                hint,
            })
        })
        .collect()
}

/// The fused cycle-major commit consumer: every column's in-progress
/// commitment, advanced per row window.
struct FusedColumns<'a, F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment> {
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

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment>
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
                    finish_streamed::<PCS>(partial, setup)
                }
                ColumnCommitState::OneHot {
                    chunk_commitments, ..
                } => finish_streamed_one_hot::<PCS>(setup, one_hot_k, &chunk_commitments),
            })
            .collect()
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F> + ModeStreamingCommitment> StreamConsumer
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

impl<F: JoltField> MaterializedColumn<F> {
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
            table: unsafe_allocate_zero_vec(table_len),
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

impl<F: JoltField> StreamConsumer for MaterializedColumn<F> {
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

#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
mod field_inline_tests {
    #![expect(clippy::unwrap_used, reason = "test module")]

    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
    };
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_openings::StreamingCommitment;
    use jolt_witness::JoltWitnessOracle;

    use super::{commit_field_inline_columns, finish_streamed};
    use crate::commitment::CommitmentGrid;
    use crate::optimized::field_registers_testing::structured_fr_fixture;

    /// The streamed FR column commit equals the commit of the explicitly
    /// laid-out table in both trace orders: cycle-major, the `T`-entry column
    /// itself (no grid padding, like the jolt increment columns);
    /// address-major, the full grid with cycle `t` at index `t · cycle_stride`
    /// and zero elsewhere.
    #[test]
    fn field_inline_commit_matches_the_dense_grid_layout() {
        let log_t = 4;
        structured_fr_fixture(12).with_plane(log_t, |backend| {
            let values: Vec<Fr> = JoltWitnessOracle::<Fr>::field_inline(backend)
                .unwrap()
                .oracle_table(FieldInlinePolynomialId::Committed(
                    FieldInlineCommittedPolynomial::FieldRdInc,
                ))
                .unwrap();
            assert!(values.iter().any(|value| *value != Fr::from_u64(0)));
            for order in [
                TracePolynomialOrder::CycleMajor,
                TracePolynomialOrder::AddressMajor,
            ] {
                let grid = CommitmentGrid {
                    total_vars: 3 + log_t,
                    log_t,
                    log_k_chunk: 3,
                    order,
                };
                let setup = DoryScheme::setup_prover(grid.total_vars);
                let streamed = commit_field_inline_columns::<Fr, DoryScheme>(
                    backend,
                    &[FieldInlineCommittedPolynomial::FieldRdInc],
                    grid,
                    &setup,
                )
                .unwrap();

                let table = match order {
                    TracePolynomialOrder::CycleMajor => values.clone(),
                    TracePolynomialOrder::AddressMajor => {
                        let mut table = vec![Fr::from_u64(0); 1 << grid.total_vars];
                        let stride = grid.cycle_stride();
                        for (cycle, value) in values.iter().enumerate() {
                            table[cycle * stride] = *value;
                        }
                        table
                    }
                };
                let mut partial = DoryScheme::begin(&setup);
                for row in table.chunks(grid.num_columns()) {
                    DoryScheme::feed(&mut partial, row, &setup);
                }
                let (commitment, hint) = finish_streamed::<DoryScheme>(partial, &setup);
                assert_eq!(streamed.len(), 1);
                assert_eq!(streamed[0].commitment, commitment, "{order:?} commitment");
                assert_eq!(streamed[0].hint, hint, "{order:?} hint");
            }
        });
    }
}
