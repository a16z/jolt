use std::collections::HashMap;

use crate::{
    BackendError, CommitmentMode, CommitmentSlot, CommittedPolynomialOutput, StreamedWitnessChunk,
    StreamedWitnessOutput,
};
use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_openings::{CommitmentScheme, ZkStreamingCommitment};
use jolt_poly::OneHotIndexOrder;
use jolt_witness::{
    OracleRef, PolynomialBatchStream, PolynomialChunk, PolynomialStream, WitnessError,
    WitnessNamespace,
};
use rayon::prelude::*;

#[cfg(feature = "frontier-harness")]
fn record_commitment_timing(label: &'static str, time_ms: f64) {
    crate::timing::record_backend_timing(label, time_ms);
}

#[cfg(not(feature = "frontier-harness"))]
const fn record_commitment_timing(_label: &'static str, _time_ms: f64) {}

pub(super) struct CpuCommitmentResult<N: WitnessNamespace, PCS: CommitmentScheme> {
    pub(super) streamed: StreamedWitnessOutput,
    pub(super) output: CommittedPolynomialOutput<N, PCS>,
}

pub(super) struct BatchCommitmentPlanItem<N: WitnessNamespace> {
    pub(super) slot: CommitmentSlot,
    pub(super) oracle: OracleRef<N>,
    pub(super) id: N::CommittedId,
    pub(super) polynomial_rows: usize,
    pub(super) layout: CommitmentLayout,
    pub(super) mode: CommitmentMode,
}

impl<N: WitnessNamespace> Clone for BatchCommitmentPlanItem<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: WitnessNamespace> Copy for BatchCommitmentPlanItem<N> {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenseCommitmentLayout {
    Standard {
        row_width: Option<usize>,
    },
    TraceEmbedding {
        row_width: usize,
        trace_rows: usize,
        address_columns: usize,
        trace_polynomial_order: TracePolynomialOrder,
    },
}

enum CommitmentAccumulator<F, PCS: ZkStreamingCommitment<Field = F>> {
    Empty,
    Dense(DenseStreamingState<F, PCS>),
    EmbeddedDense(EmbeddedTraceDenseStreamingState<F, PCS>),
    U64(U64StreamingState<PCS>),
    I128(I128StreamingState<PCS>),
    OneHot(OneHotCommitmentState<PCS>),
}

struct BatchCommitmentState<F, PCS: ZkStreamingCommitment<Field = F>> {
    chunks: Vec<StreamedWitnessChunk>,
    layout: CommitmentLayout,
    expected_pcs_rows: usize,
    accumulator: CommitmentAccumulator<F, PCS>,
}

impl<F, PCS> BatchCommitmentState<F, PCS>
where
    F: jolt_field::Field,
    PCS: ZkStreamingCommitment<Field = F>,
{
    fn new<N: WitnessNamespace>(
        polynomial_rows: usize,
        layout: CommitmentLayout,
    ) -> Result<Self, BackendError> {
        let dense_layout = resolve_dense_layout::<N>(polynomial_rows, layout.dense_layout)?;
        let layout = CommitmentLayout {
            dense_layout,
            ..layout
        };
        let expected_pcs_rows = expected_pcs_rows::<N>(polynomial_rows, dense_layout)?;
        Ok(Self {
            chunks: Vec::new(),
            layout,
            expected_pcs_rows,
            accumulator: CommitmentAccumulator::new(),
        })
    }

    fn append<N: WitnessNamespace>(
        &mut self,
        chunk: PolynomialChunk<F>,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        let kind = chunk.kind();
        let rows = chunk.len();
        let index = self.chunks.len();
        self.chunks
            .push(StreamedWitnessChunk::new(index, kind, rows));
        self.accumulator.append::<N>(chunk, self.layout, setup)
    }

    fn finish<N: WitnessNamespace>(
        self,
        item: BatchCommitmentPlanItem<N>,
        setup: &PCS::ProverSetup,
    ) -> Result<CpuCommitmentResult<N, PCS>, BackendError> {
        let streamed = StreamedWitnessOutput::new(item.slot, self.chunks);
        let finish_label = self.accumulator.finish_timing_label();
        let start = std::time::Instant::now();
        let (commitment, opening_hint) = self.accumulator.finish::<N>(
            item.polynomial_rows,
            self.expected_pcs_rows,
            item.layout.one_hot_index_order,
            item.mode,
            setup,
        )?;
        record_commitment_timing(finish_label, start.elapsed().as_secs_f64() * 1000.0);
        Ok(CpuCommitmentResult {
            streamed,
            output: CommittedPolynomialOutput::new(
                item.slot,
                item.oracle,
                item.polynomial_rows,
                commitment,
                opening_hint,
            ),
        })
    }
}

impl<F, PCS> CommitmentAccumulator<F, PCS>
where
    F: jolt_field::Field,
    PCS: ZkStreamingCommitment<Field = F>,
{
    const fn new() -> Self {
        Self::Empty
    }

    const fn finish_timing_label(&self) -> &'static str {
        match self {
            Self::Empty => "stage0.backend.batch.finish.empty",
            Self::Dense(_) => "stage0.backend.batch.finish.dense",
            Self::EmbeddedDense(_) => "stage0.backend.batch.finish.embedded_dense",
            Self::U64(_) => "stage0.backend.batch.finish.u64",
            Self::I128(_) => "stage0.backend.batch.finish.i128",
            Self::OneHot(_) => "stage0.backend.batch.finish.one_hot",
        }
    }

    fn append<N: WitnessNamespace>(
        &mut self,
        chunk: PolynomialChunk<F>,
        layout: CommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match chunk {
            PolynomialChunk::Dense(values) => {
                self.append_dense::<N>(values, layout.dense_layout, setup)
            }
            PolynomialChunk::Zeros(rows) => self.append_zeros::<N>(rows, layout, setup),
            PolynomialChunk::U8(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u8).collect(),
                layout.dense_layout,
                setup,
            ),
            PolynomialChunk::U16(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u16).collect(),
                layout.dense_layout,
                setup,
            ),
            PolynomialChunk::U32(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u32).collect(),
                layout.dense_layout,
                setup,
            ),
            PolynomialChunk::U64(values) => {
                self.append_u64::<N>(values, layout.dense_layout, setup)
            }
            PolynomialChunk::I64(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_i64).collect(),
                layout.dense_layout,
                setup,
            ),
            PolynomialChunk::I128(values) => {
                self.append_i128::<N>(values, layout.dense_layout, setup)
            }
            PolynomialChunk::OneHot(values) => self.append_one_hot::<N>(values, layout, setup),
        }
    }

    fn append_zeros<N: WitnessNamespace>(
        &mut self,
        rows: usize,
        layout: CommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::Dense(existing) => existing.append_zeros(rows, setup),
            Self::EmbeddedDense(existing) => existing.append_zeros(rows, setup),
            Self::U64(existing) => existing.append_zeros(rows, setup),
            Self::I128(existing) => existing.append_zeros(rows, setup),
            Self::OneHot(existing) => existing.append_zeros::<N>(rows, setup)?,
            Self::Empty => match layout.dense_layout {
                DenseCommitmentLayout::Standard { row_width } => {
                    let row_width = row_width.ok_or_else(|| WitnessError::InvalidDimensions {
                        namespace: N::ID.name,
                        reason: "standard zero commitment layout missing row width".to_owned(),
                    })?;
                    let mut state = DenseStreamingState::<F, PCS>::new(row_width, setup);
                    state.append_zeros(rows, setup);
                    *self = Self::Dense(state);
                }
                DenseCommitmentLayout::TraceEmbedding {
                    row_width,
                    trace_rows,
                    address_columns,
                    trace_polynomial_order,
                } => {
                    let mut state = EmbeddedTraceDenseStreamingState::<F, PCS>::new(
                        row_width,
                        trace_rows,
                        address_columns,
                        trace_polynomial_order,
                        setup,
                    );
                    state.append_zeros(rows, setup);
                    *self = Self::EmbeddedDense(state);
                }
            },
        }
        Ok(())
    }

    fn finish<N: WitnessNamespace>(
        self,
        polynomial_rows: usize,
        expected_pcs_rows: usize,
        one_hot_index_order: OneHotIndexOrder,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        match self {
            Self::Empty => Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: "cannot commit an empty witness stream".to_owned(),
            }
            .into()),
            Self::Dense(state) => state.finish::<N>(expected_pcs_rows, mode, setup),
            Self::EmbeddedDense(state) => state.finish::<N>(expected_pcs_rows, mode, setup),
            Self::U64(state) => state.finish::<N>(expected_pcs_rows, mode, setup),
            Self::I128(state) => state.finish::<N>(expected_pcs_rows, mode, setup),
            Self::OneHot(state) => {
                let _ = expected_pcs_rows;
                state.finish::<N>(polynomial_rows, one_hot_index_order, mode, setup)
            }
        }
    }

    fn append_dense<N: WitnessNamespace>(
        &mut self,
        values: Vec<F>,
        dense_layout: DenseCommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::Dense(existing) => existing.append_values(values, setup),
            Self::EmbeddedDense(existing) => existing.append_values(values, setup),
            Self::U64(_) | Self::I128(_) | Self::OneHot(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "dense chunk followed a differently encoded chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => match dense_layout {
                DenseCommitmentLayout::Standard { row_width } => {
                    let row_width = row_width.ok_or_else(|| WitnessError::InvalidDimensions {
                        namespace: N::ID.name,
                        reason: "standard dense commitment layout missing row width".to_owned(),
                    })?;
                    let mut state = DenseStreamingState::<F, PCS>::new(row_width, setup);
                    state.append_values(values, setup);
                    *self = Self::Dense(state);
                }
                DenseCommitmentLayout::TraceEmbedding {
                    row_width,
                    trace_rows,
                    address_columns,
                    trace_polynomial_order,
                } => {
                    let mut state = EmbeddedTraceDenseStreamingState::<F, PCS>::new(
                        row_width,
                        trace_rows,
                        address_columns,
                        trace_polynomial_order,
                        setup,
                    );
                    state.append_values(values, setup);
                    *self = Self::EmbeddedDense(state);
                }
            },
        }
        Ok(())
    }

    fn append_embedded_scalar_values<N: WitnessNamespace>(
        &mut self,
        values: impl IntoIterator<Item = F>,
        dense_layout: DenseCommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::EmbeddedDense(existing) => existing.append_values(values, setup),
            Self::Dense(_) | Self::U64(_) | Self::I128(_) | Self::OneHot(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "embedded dense chunk followed a differently encoded chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => {
                let DenseCommitmentLayout::TraceEmbedding {
                    row_width,
                    trace_rows,
                    address_columns,
                    trace_polynomial_order,
                } = dense_layout
                else {
                    return Err(WitnessError::InvalidDimensions {
                        namespace: N::ID.name,
                        reason: "embedded scalar append requires trace embedding layout".to_owned(),
                    }
                    .into());
                };
                let mut state = EmbeddedTraceDenseStreamingState::<F, PCS>::new(
                    row_width,
                    trace_rows,
                    address_columns,
                    trace_polynomial_order,
                    setup,
                );
                state.append_values(values, setup);
                *self = Self::EmbeddedDense(state);
            }
        }
        Ok(())
    }

    fn append_u64<N: WitnessNamespace>(
        &mut self,
        values: Vec<u64>,
        dense_layout: DenseCommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        if matches!(dense_layout, DenseCommitmentLayout::TraceEmbedding { .. }) {
            return self.append_embedded_scalar_values::<N>(
                values.into_iter().map(F::from_u64),
                dense_layout,
                setup,
            );
        }
        let DenseCommitmentLayout::Standard {
            row_width: Some(row_width),
        } = dense_layout
        else {
            return Err(WitnessError::InvalidDimensions {
                namespace: N::ID.name,
                reason: "u64 commitment layout missing row width".to_owned(),
            }
            .into());
        };
        match self {
            Self::U64(existing) => existing.append_values(values, setup),
            Self::Dense(_) | Self::EmbeddedDense(_) | Self::I128(_) | Self::OneHot(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "u64 chunk followed a differently encoded chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => {
                let mut state = U64StreamingState::<PCS>::new(row_width, setup);
                state.append_values(values, setup);
                *self = Self::U64(state);
            }
        }
        Ok(())
    }

    fn append_i128<N: WitnessNamespace>(
        &mut self,
        values: Vec<i128>,
        dense_layout: DenseCommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        if matches!(dense_layout, DenseCommitmentLayout::TraceEmbedding { .. }) {
            return self.append_embedded_scalar_values::<N>(
                values.into_iter().map(F::from_i128),
                dense_layout,
                setup,
            );
        }
        let DenseCommitmentLayout::Standard {
            row_width: Some(row_width),
        } = dense_layout
        else {
            return Err(WitnessError::InvalidDimensions {
                namespace: N::ID.name,
                reason: "i128 commitment layout missing row width".to_owned(),
            }
            .into());
        };
        match self {
            Self::I128(existing) => existing.append_values(values, setup),
            Self::Dense(_) | Self::EmbeddedDense(_) | Self::U64(_) | Self::OneHot(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "i128 chunk followed a differently encoded chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => {
                let mut state = I128StreamingState::<PCS>::new(row_width, setup);
                state.append_values(values, setup);
                *self = Self::I128(state);
            }
        }
        Ok(())
    }

    fn append_one_hot<N: WitnessNamespace>(
        &mut self,
        values: Vec<Option<usize>>,
        layout: CommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::OneHot(existing) => existing.append::<N>(values, setup)?,
            Self::Dense(_) | Self::EmbeddedDense(_) | Self::U64(_) | Self::I128(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "one-hot chunk followed a differently encoded chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => {
                *self = Self::OneHot(OneHotCommitmentState::new::<N>(values, layout, setup)?);
            }
        }
        Ok(())
    }
}

enum OneHotCommitmentState<PCS: ZkStreamingCommitment> {
    Materialized(Vec<Option<u8>>),
    ColumnMajor(OneHotColumnMajorStreamingState<PCS>),
}

struct OneHotColumnMajorStreamingState<PCS: ZkStreamingCommitment> {
    chunks: Vec<PCS::OneHotChunkCommitment>,
    context: PCS::OneHotStreamContext,
    row_width: usize,
    trace_rows: usize,
    one_hot_k: usize,
    rows_seen: usize,
}

impl<PCS> OneHotCommitmentState<PCS>
where
    PCS: ZkStreamingCommitment,
{
    fn new<N: WitnessNamespace>(
        values: Vec<Option<usize>>,
        layout: CommitmentLayout,
        setup: &PCS::ProverSetup,
    ) -> Result<Self, BackendError> {
        match layout.one_hot_streaming {
            Some(streaming) => {
                let mut state = OneHotColumnMajorStreamingState::<PCS>::new(streaming, setup);
                state.append::<N>(values, setup)?;
                Ok(Self::ColumnMajor(state))
            }
            None => Ok(Self::Materialized(materialized_one_hot_values::<N>(
                values,
            )?)),
        }
    }

    fn append<N: WitnessNamespace>(
        &mut self,
        values: Vec<Option<usize>>,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::Materialized(existing) => {
                existing.extend(materialized_one_hot_values::<N>(values)?);
                Ok(())
            }
            Self::ColumnMajor(existing) => existing.append::<N>(values, setup),
        }
    }

    fn append_zeros<N: WitnessNamespace>(
        &mut self,
        rows: usize,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::Materialized(existing) => {
                existing.extend(std::iter::repeat_n(None, rows));
                Ok(())
            }
            Self::ColumnMajor(existing) => {
                existing.append::<N>(std::iter::repeat_n(None, rows).collect(), setup)
            }
        }
    }

    fn finish<N: WitnessNamespace>(
        self,
        polynomial_rows: usize,
        index_order: OneHotIndexOrder,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        match self {
            Self::Materialized(indices) => {
                commit_one_hot::<PCS, N>(indices, polynomial_rows, index_order, mode, setup)
            }
            Self::ColumnMajor(state) => state.finish::<N>(polynomial_rows, mode, setup),
        }
    }
}

impl<PCS> OneHotColumnMajorStreamingState<PCS>
where
    PCS: ZkStreamingCommitment,
{
    fn new(layout: OneHotStreamingLayout, setup: &PCS::ProverSetup) -> Self {
        Self {
            chunks: Vec::new(),
            context: PCS::begin_one_hot_column_major_stream(setup, layout.row_width),
            row_width: layout.row_width,
            trace_rows: layout.trace_rows,
            one_hot_k: layout.one_hot_k,
            rows_seen: 0,
        }
    }

    fn append<N: WitnessNamespace>(
        &mut self,
        values: Vec<Option<usize>>,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        if values.len() != self.row_width {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "column-major one-hot chunk has {} rows, expected {}",
                    values.len(),
                    self.row_width
                ),
            }
            .into());
        }
        if self.rows_seen + values.len() > self.trace_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "column-major one-hot stream exceeded trace rows: {} + {} > {}",
                    self.rows_seen,
                    values.len(),
                    self.trace_rows
                ),
            }
            .into());
        }
        if let Some(value) = values
            .iter()
            .flatten()
            .copied()
            .find(|&value| value >= self.one_hot_k)
        {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!("one-hot column {value} is outside k={}", self.one_hot_k),
            }
            .into());
        }

        self.chunks.push(PCS::process_one_hot_chunk(
            &mut self.context,
            setup,
            self.one_hot_k,
            &values,
        ));
        self.rows_seen += values.len();
        Ok(())
    }

    fn finish<N: WitnessNamespace>(
        self,
        polynomial_rows: usize,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        if self.rows_seen != self.trace_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "column-major one-hot stream produced {} trace rows, expected {}",
                    self.rows_seen, self.trace_rows
                ),
            }
            .into());
        }
        if self.trace_rows.checked_mul(self.one_hot_k) != Some(polynomial_rows) {
            return Err(WitnessError::InvalidDimensions {
                namespace: N::ID.name,
                reason: format!(
                    "column-major one-hot shape T={} K={} does not match descriptor rows {polynomial_rows}",
                    self.trace_rows, self.one_hot_k
                ),
            }
            .into());
        }
        Ok(match mode {
            CommitmentMode::Transparent => {
                PCS::finish_one_hot_column_major_chunks(setup, self.one_hot_k, &self.chunks)
            }
            CommitmentMode::Zk => {
                PCS::finish_zk_one_hot_column_major_chunks(setup, self.one_hot_k, &self.chunks)
            }
        })
    }
}

fn materialized_one_hot_values<N: WitnessNamespace>(
    values: Vec<Option<usize>>,
) -> Result<Vec<Option<u8>>, BackendError> {
    values
        .into_iter()
        .map(|value| {
            value
                .map(|value| {
                    u8::try_from(value).map_err(|_| WitnessError::InvalidWitnessData {
                        namespace: N::ID.name,
                        reason: format!("one-hot column {value} is outside u8 range"),
                    })
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

struct EmbeddedTraceDenseStreamingState<F, PCS: ZkStreamingCommitment<Field = F>> {
    partial: PCS::PartialCommitment,
    row_buffer: Vec<F>,
    row_width: usize,
    trace_rows: usize,
    address_columns: usize,
    trace_polynomial_order: TracePolynomialOrder,
    current_row: usize,
    emitted_trace_values: usize,
    rows_fed: usize,
}

impl<F, PCS> EmbeddedTraceDenseStreamingState<F, PCS>
where
    F: jolt_field::Field,
    PCS: ZkStreamingCommitment<Field = F>,
{
    fn new(
        row_width: usize,
        trace_rows: usize,
        address_columns: usize,
        trace_polynomial_order: TracePolynomialOrder,
        setup: &PCS::ProverSetup,
    ) -> Self {
        Self {
            partial: PCS::begin(setup),
            row_buffer: vec![F::zero(); row_width],
            row_width,
            trace_rows,
            address_columns,
            trace_polynomial_order,
            current_row: 0,
            emitted_trace_values: 0,
            rows_fed: 0,
        }
    }

    fn append_values(&mut self, values: impl IntoIterator<Item = F>, setup: &PCS::ProverSetup) {
        for value in values {
            let flat = self.trace_polynomial_order.address_cycle_to_index(
                0,
                self.emitted_trace_values,
                self.address_columns,
                self.trace_rows,
            );
            let target_row = flat / self.row_width;
            let target_col = flat % self.row_width;
            while self.current_row < target_row {
                self.feed_current_row(setup);
            }
            self.row_buffer[target_col] = value;
            self.emitted_trace_values += 1;
        }
    }

    fn append_zeros(&mut self, rows: usize, setup: &PCS::ProverSetup) {
        for _ in 0..rows {
            let flat = self.trace_polynomial_order.address_cycle_to_index(
                0,
                self.emitted_trace_values,
                self.address_columns,
                self.trace_rows,
            );
            let target_row = flat / self.row_width;
            while self.current_row < target_row {
                self.feed_current_row(setup);
            }
            self.emitted_trace_values += 1;
        }
    }

    fn feed_current_row(&mut self, setup: &PCS::ProverSetup) {
        if self.row_buffer.iter().all(F::is_zero) {
            PCS::feed_zeros(&mut self.partial, self.row_width, 1, setup);
        } else {
            PCS::feed(&mut self.partial, &self.row_buffer, setup);
        }
        self.row_buffer.fill(F::zero());
        self.current_row += 1;
        self.rows_fed += 1;
    }

    fn finish<N: WitnessNamespace>(
        mut self,
        expected_rows: usize,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        if self.emitted_trace_values != self.trace_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "embedded dense stream produced {} trace rows, expected {}",
                    self.emitted_trace_values, self.trace_rows,
                ),
            }
            .into());
        }
        while self.rows_fed < expected_rows {
            if self.row_buffer.iter().all(F::is_zero) {
                let remaining = expected_rows - self.rows_fed;
                PCS::feed_zeros(&mut self.partial, self.row_width, remaining, setup);
                self.current_row += remaining;
                self.rows_fed = expected_rows;
                break;
            }
            self.feed_current_row(setup);
        }
        Ok(finish_with_mode::<PCS>(self.partial, mode, setup))
    }
}

struct DenseStreamingState<F, PCS: ZkStreamingCommitment<Field = F>> {
    partial: PCS::PartialCommitment,
    row_buffer: Vec<F>,
    rows_fed: usize,
    row_width: usize,
}

impl<F, PCS> DenseStreamingState<F, PCS>
where
    F: jolt_field::Field,
    PCS: ZkStreamingCommitment<Field = F>,
{
    fn new(row_width: usize, setup: &PCS::ProverSetup) -> Self {
        Self {
            partial: PCS::begin(setup),
            row_buffer: Vec::with_capacity(row_width),
            rows_fed: 0,
            row_width,
        }
    }

    fn append_values(&mut self, values: Vec<F>, setup: &PCS::ProverSetup) {
        if self.row_buffer.is_empty() && values.len().is_multiple_of(self.row_width) {
            for row in values.chunks(self.row_width) {
                PCS::feed(&mut self.partial, row, setup);
                self.rows_fed += 1;
            }
            return;
        }

        for value in values {
            self.row_buffer.push(value);
            if self.row_buffer.len() == self.row_width {
                PCS::feed(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn append_zeros(&mut self, values: usize, setup: &PCS::ProverSetup) {
        if values == 0 {
            return;
        }
        if self.row_buffer.is_empty() {
            let full_rows = values / self.row_width;
            if full_rows != 0 {
                PCS::feed_zeros(&mut self.partial, self.row_width, full_rows, setup);
                self.rows_fed += full_rows;
            }
            let remainder = values % self.row_width;
            self.row_buffer.resize(remainder, F::zero());
            return;
        }

        for _ in 0..values {
            self.row_buffer.push(F::zero());
            if self.row_buffer.len() == self.row_width {
                PCS::feed(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn finish<N: WitnessNamespace>(
        self,
        expected_rows: usize,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        if !self.row_buffer.is_empty() {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "dense stream ended with partial PCS row of {} values; row width is {}",
                    self.row_buffer.len(),
                    self.row_width,
                ),
            }
            .into());
        }
        if self.rows_fed != expected_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "dense stream produced {} PCS rows, expected {expected_rows}",
                    self.rows_fed,
                ),
            }
            .into());
        }
        Ok(finish_with_mode::<PCS>(self.partial, mode, setup))
    }
}

struct U64StreamingState<PCS: ZkStreamingCommitment> {
    partial: PCS::PartialCommitment,
    row_buffer: Vec<u64>,
    rows_fed: usize,
    row_width: usize,
}

impl<PCS> U64StreamingState<PCS>
where
    PCS: ZkStreamingCommitment,
{
    fn new(row_width: usize, setup: &PCS::ProverSetup) -> Self {
        Self {
            partial: PCS::begin(setup),
            row_buffer: Vec::with_capacity(row_width),
            rows_fed: 0,
            row_width,
        }
    }

    fn append_values(&mut self, values: Vec<u64>, setup: &PCS::ProverSetup) {
        if self.row_buffer.is_empty() && values.len().is_multiple_of(self.row_width) {
            for row in values.chunks(self.row_width) {
                PCS::feed_u64(&mut self.partial, row, setup);
                self.rows_fed += 1;
            }
            return;
        }

        for value in values {
            self.row_buffer.push(value);
            if self.row_buffer.len() == self.row_width {
                PCS::feed_u64(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn append_zeros(&mut self, values: usize, setup: &PCS::ProverSetup) {
        if values == 0 {
            return;
        }
        if self.row_buffer.is_empty() {
            let full_rows = values / self.row_width;
            if full_rows != 0 {
                PCS::feed_zeros(&mut self.partial, self.row_width, full_rows, setup);
                self.rows_fed += full_rows;
            }
            let remainder = values % self.row_width;
            self.row_buffer.resize(remainder, 0);
            return;
        }

        for _ in 0..values {
            self.row_buffer.push(0);
            if self.row_buffer.len() == self.row_width {
                PCS::feed_u64(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn finish<N: WitnessNamespace>(
        self,
        expected_rows: usize,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        if !self.row_buffer.is_empty() {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "u64 stream ended with partial PCS row of {} values; row width is {}",
                    self.row_buffer.len(),
                    self.row_width,
                ),
            }
            .into());
        }
        if self.rows_fed != expected_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "u64 stream produced {} PCS rows, expected {expected_rows}",
                    self.rows_fed,
                ),
            }
            .into());
        }
        Ok(finish_with_mode::<PCS>(self.partial, mode, setup))
    }
}

struct I128StreamingState<PCS: ZkStreamingCommitment> {
    partial: PCS::PartialCommitment,
    row_buffer: Vec<i128>,
    rows_fed: usize,
    row_width: usize,
}

impl<PCS> I128StreamingState<PCS>
where
    PCS: ZkStreamingCommitment,
{
    fn new(row_width: usize, setup: &PCS::ProverSetup) -> Self {
        Self {
            partial: PCS::begin(setup),
            row_buffer: Vec::with_capacity(row_width),
            rows_fed: 0,
            row_width,
        }
    }

    fn append_values(&mut self, values: Vec<i128>, setup: &PCS::ProverSetup) {
        if self.row_buffer.is_empty() && values.len().is_multiple_of(self.row_width) {
            for row in values.chunks(self.row_width) {
                PCS::feed_i128(&mut self.partial, row, setup);
                self.rows_fed += 1;
            }
            return;
        }

        for value in values {
            self.row_buffer.push(value);
            if self.row_buffer.len() == self.row_width {
                PCS::feed_i128(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn append_zeros(&mut self, values: usize, setup: &PCS::ProverSetup) {
        if values == 0 {
            return;
        }
        if self.row_buffer.is_empty() {
            let full_rows = values / self.row_width;
            if full_rows != 0 {
                PCS::feed_zeros(&mut self.partial, self.row_width, full_rows, setup);
                self.rows_fed += full_rows;
            }
            let remainder = values % self.row_width;
            self.row_buffer.resize(remainder, 0);
            return;
        }

        for _ in 0..values {
            self.row_buffer.push(0);
            if self.row_buffer.len() == self.row_width {
                PCS::feed_i128(&mut self.partial, &self.row_buffer, setup);
                self.row_buffer.clear();
                self.rows_fed += 1;
            }
        }
    }

    fn finish<N: WitnessNamespace>(
        self,
        expected_rows: usize,
        mode: CommitmentMode,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        if !self.row_buffer.is_empty() {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "i128 stream ended with partial PCS row of {} values; row width is {}",
                    self.row_buffer.len(),
                    self.row_width,
                ),
            }
            .into());
        }
        if self.rows_fed != expected_rows {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!(
                    "i128 stream produced {} PCS rows, expected {expected_rows}",
                    self.rows_fed,
                ),
            }
            .into());
        }
        Ok(finish_with_mode::<PCS>(self.partial, mode, setup))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CommitmentLayout {
    pub(super) dense_layout: DenseCommitmentLayout,
    pub(super) one_hot_index_order: OneHotIndexOrder,
    pub(super) one_hot_streaming: Option<OneHotStreamingLayout>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OneHotStreamingLayout {
    pub(super) row_width: usize,
    pub(super) trace_rows: usize,
    pub(super) one_hot_k: usize,
}

impl CommitmentLayout {
    pub(super) const fn default_for_request() -> Self {
        Self {
            dense_layout: DenseCommitmentLayout::Standard { row_width: None },
            one_hot_index_order: OneHotIndexOrder::RowMajor,
            one_hot_streaming: None,
        }
    }
}

pub(super) fn commit_streamed_witness<F, PCS, N>(
    slot: CommitmentSlot,
    oracle: jolt_witness::OracleRef<N>,
    polynomial_rows: usize,
    layout: CommitmentLayout,
    mode: CommitmentMode,
    stream: &mut dyn PolynomialStream<F>,
    setup: &PCS::ProverSetup,
) -> Result<CpuCommitmentResult<N, PCS>, BackendError>
where
    F: jolt_field::Field,
    N: WitnessNamespace + Send + Sync,
    N::CommittedId: Send + Sync,
    N::VirtualId: Send + Sync,
    PCS: ZkStreamingCommitment<Field = F>,
    PCS::OpeningHint: Send,
    PCS::Output: Send,
    PCS::PartialCommitment: Send,
    PCS::ProverSetup: Sync,
{
    let mut chunks = Vec::new();
    let dense_layout = resolve_dense_layout::<N>(polynomial_rows, layout.dense_layout)?;
    let expected_pcs_rows = expected_pcs_rows::<N>(polynomial_rows, dense_layout)?;
    let layout = CommitmentLayout {
        dense_layout,
        ..layout
    };
    let mut accumulator = CommitmentAccumulator::<F, PCS>::new();

    while let Some(chunk) = stream.next_chunk()? {
        let kind = chunk.kind();
        let index = chunks.len();
        chunks.push(StreamedWitnessChunk::new(index, kind, chunk.len()));
        accumulator.append::<N>(chunk, layout, setup)?;
    }

    let streamed = StreamedWitnessOutput::new(slot, chunks);
    let (commitment, opening_hint) = accumulator.finish::<N>(
        polynomial_rows,
        expected_pcs_rows,
        layout.one_hot_index_order,
        mode,
        setup,
    )?;

    Ok(CpuCommitmentResult {
        streamed,
        output: CommittedPolynomialOutput::new(
            slot,
            oracle,
            polynomial_rows,
            commitment,
            opening_hint,
        ),
    })
}

pub(super) fn commit_batched_streamed_witness<F, PCS, N>(
    items: &[BatchCommitmentPlanItem<N>],
    stream: &mut dyn PolynomialBatchStream<F, N>,
    setup: &PCS::ProverSetup,
) -> Result<Vec<CpuCommitmentResult<N, PCS>>, BackendError>
where
    F: jolt_field::Field,
    N: WitnessNamespace + Send + Sync,
    N::CommittedId: Send + Sync,
    N::VirtualId: Send + Sync,
    PCS: ZkStreamingCommitment<Field = F>,
    PCS::OpeningHint: Send,
    PCS::Output: Send,
    PCS::PartialCommitment: Send,
    PCS::ProverSetup: Sync,
{
    let start = std::time::Instant::now();
    let mut id_to_index = HashMap::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        if id_to_index.insert(item.id, index).is_some() {
            return Err(BackendError::InvalidRequest {
                backend: "cpu",
                task: "batched streamed witness commitment",
                reason: format!("duplicate committed oracle in batch: {:?}", item.id),
            });
        }
    }
    record_commitment_timing(
        "stage0.backend.batch.index_items",
        start.elapsed().as_secs_f64() * 1000.0,
    );

    let start = std::time::Instant::now();
    let mut states = items
        .iter()
        .map(|item| BatchCommitmentState::<F, PCS>::new::<N>(item.polynomial_rows, item.layout))
        .collect::<Result<Vec<_>, _>>()?;
    record_commitment_timing(
        "stage0.backend.batch.init_states",
        start.elapsed().as_secs_f64() * 1000.0,
    );

    let mut next_batch_ms = 0.0;
    let mut order_chunks_ms = 0.0;
    let mut append_chunks_ms = 0.0;
    loop {
        let start = std::time::Instant::now();
        let next = stream.next_batch()?;
        next_batch_ms += start.elapsed().as_secs_f64() * 1000.0;
        let Some(next) = next else {
            break;
        };
        let start = std::time::Instant::now();
        let chunks_by_item = ordered_batch_chunks(next.chunks, items, &id_to_index)?;
        order_chunks_ms += start.elapsed().as_secs_f64() * 1000.0;
        let start = std::time::Instant::now();
        states
            .par_iter_mut()
            .zip(chunks_by_item.into_par_iter())
            .try_for_each(|(state, chunk)| state.append::<N>(chunk, setup))?;
        append_chunks_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
    record_commitment_timing("stage0.backend.batch.next_batch", next_batch_ms);
    record_commitment_timing("stage0.backend.batch.order_chunks", order_chunks_ms);
    record_commitment_timing("stage0.backend.batch.append_chunks", append_chunks_ms);

    let start = std::time::Instant::now();
    items
        .iter()
        .copied()
        .zip(states)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(item, state)| state.finish::<N>(item, setup))
        .collect::<Result<Vec<_>, _>>()
        .inspect(|_| {
            record_commitment_timing(
                "stage0.backend.batch.finish_states",
                start.elapsed().as_secs_f64() * 1000.0,
            );
        })
}

fn ordered_batch_chunks<N: WitnessNamespace, F>(
    chunks: Vec<(N::CommittedId, PolynomialChunk<F>)>,
    items: &[BatchCommitmentPlanItem<N>],
    id_to_index: &HashMap<N::CommittedId, usize>,
) -> Result<Vec<PolynomialChunk<F>>, BackendError>
where
    N::CommittedId: Send + Sync,
{
    if chunks.len() != items.len() {
        return Err(WitnessError::InvalidWitnessData {
            namespace: N::ID.name,
            reason: format!(
                "batched stream emitted {} chunks for {} requested oracles",
                chunks.len(),
                items.len(),
            ),
        }
        .into());
    }

    if chunks
        .iter()
        .zip(items.iter())
        .all(|((id, _), item)| *id == item.id)
    {
        return Ok(chunks.into_iter().map(|(_, chunk)| chunk).collect());
    }

    let mut chunks_by_item = (0..items.len()).map(|_| None).collect::<Vec<_>>();
    for (id, chunk) in chunks {
        let Some(&index) = id_to_index.get(&id) else {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!("batched stream emitted unrequested oracle {id:?}"),
            }
            .into());
        };
        if chunks_by_item[index].replace(chunk).is_some() {
            return Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: format!("batched stream emitted duplicate oracle {id:?}"),
            }
            .into());
        }
    }

    chunks_by_item
        .into_iter()
        .zip(items.iter())
        .map(|(chunk, item)| {
            chunk.ok_or_else(|| {
                WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: format!("batched stream omitted requested oracle {:?}", item.id),
                }
                .into()
            })
        })
        .collect()
}

fn resolve_dense_layout<N: WitnessNamespace>(
    polynomial_rows: usize,
    dense_layout: DenseCommitmentLayout,
) -> Result<DenseCommitmentLayout, BackendError> {
    match dense_layout {
        DenseCommitmentLayout::Standard { row_width } => {
            let row_width = row_width.map_or_else(|| pcs_row_width::<N>(polynomial_rows), Ok)?;
            validate_row_width::<N>(polynomial_rows, row_width)?;
            Ok(DenseCommitmentLayout::Standard {
                row_width: Some(row_width),
            })
        }
        DenseCommitmentLayout::TraceEmbedding {
            row_width,
            trace_rows,
            address_columns,
            trace_polynomial_order,
        } => {
            if polynomial_rows != trace_rows {
                return Err(WitnessError::InvalidDimensions {
                    namespace: N::ID.name,
                    reason: format!(
                        "embedded trace polynomial rows {trace_rows} must match descriptor rows {polynomial_rows}",
                    ),
                }
                .into());
            }
            let full_rows = trace_rows.checked_mul(address_columns).ok_or_else(|| {
                WitnessError::InvalidDimensions {
                    namespace: N::ID.name,
                    reason: "embedded trace polynomial dimensions overflowed".to_owned(),
                }
            })?;
            validate_row_width::<N>(full_rows, row_width)?;
            Ok(DenseCommitmentLayout::TraceEmbedding {
                row_width,
                trace_rows,
                address_columns,
                trace_polynomial_order,
            })
        }
    }
}

fn expected_pcs_rows<N: WitnessNamespace>(
    polynomial_rows: usize,
    dense_layout: DenseCommitmentLayout,
) -> Result<usize, BackendError> {
    match dense_layout {
        DenseCommitmentLayout::Standard {
            row_width: Some(row_width),
        } => Ok(polynomial_rows / row_width),
        DenseCommitmentLayout::TraceEmbedding {
            row_width,
            trace_rows,
            address_columns,
            ..
        } => Ok(trace_rows * address_columns / row_width),
        DenseCommitmentLayout::Standard { row_width: None } => {
            Err(WitnessError::InvalidDimensions {
                namespace: N::ID.name,
                reason: "standard dense commitment layout missing row width".to_owned(),
            }
            .into())
        }
    }
}

fn pcs_row_width<N: WitnessNamespace>(polynomial_rows: usize) -> Result<usize, BackendError> {
    if !polynomial_rows.is_power_of_two() {
        return Err(WitnessError::InvalidDimensions {
            namespace: N::ID.name,
            reason: format!("committed polynomial rows must be a power of two: {polynomial_rows}"),
        }
        .into());
    }
    let log_rows = polynomial_rows.trailing_zeros() as usize;
    Ok(1_usize << log_rows.div_ceil(2))
}

fn validate_row_width<N: WitnessNamespace>(
    polynomial_rows: usize,
    row_width: usize,
) -> Result<(), BackendError> {
    if row_width == 0 || row_width > polynomial_rows || !polynomial_rows.is_multiple_of(row_width) {
        return Err(WitnessError::InvalidDimensions {
            namespace: N::ID.name,
            reason: format!(
                "PCS row width {row_width} must evenly divide polynomial rows {polynomial_rows}",
            ),
        }
        .into());
    }
    Ok(())
}

fn commit_one_hot<PCS, N: WitnessNamespace>(
    indices: Vec<Option<u8>>,
    polynomial_rows: usize,
    index_order: OneHotIndexOrder,
    mode: CommitmentMode,
    setup: &PCS::ProverSetup,
) -> Result<(PCS::Output, PCS::OpeningHint), BackendError>
where
    PCS: ZkStreamingCommitment,
{
    if indices.is_empty() || !polynomial_rows.is_multiple_of(indices.len()) {
        return Err(WitnessError::InvalidWitnessData {
            namespace: N::ID.name,
            reason: format!(
                "one-hot stream with {} rows cannot fill descriptor rows {polynomial_rows}",
                indices.len(),
            ),
        }
        .into());
    }
    let k = polynomial_rows / indices.len();
    if let Some(value) = indices
        .iter()
        .flatten()
        .copied()
        .find(|&value| value as usize >= k)
    {
        return Err(WitnessError::InvalidWitnessData {
            namespace: N::ID.name,
            reason: format!("one-hot column {value} is outside k={k}"),
        }
        .into());
    }
    let poly = jolt_poly::OneHotPolynomial::new_with_index_order(k, indices, index_order);
    Ok(match mode {
        CommitmentMode::Transparent => PCS::commit(&poly, setup),
        CommitmentMode::Zk => PCS::commit_zk(&poly, setup),
    })
}

fn finish_with_mode<PCS>(
    partial: PCS::PartialCommitment,
    mode: CommitmentMode,
    setup: &PCS::ProverSetup,
) -> (PCS::Output, PCS::OpeningHint)
where
    PCS: ZkStreamingCommitment,
{
    match mode {
        CommitmentMode::Transparent => PCS::finish_with_hint(partial, setup),
        CommitmentMode::Zk => PCS::finish_zk_with_hint(partial, setup),
    }
}
