use crate::{
    BackendError, CommitmentSlot, CommittedPolynomialOutput, StreamedWitnessChunk,
    StreamedWitnessOutput,
};
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::{PolynomialChunk, PolynomialStream, WitnessError, WitnessNamespace};

pub(super) struct CpuCommitmentResult<N: WitnessNamespace, PCS: CommitmentScheme> {
    pub(super) streamed: StreamedWitnessOutput,
    pub(super) output: CommittedPolynomialOutput<N, PCS>,
}

enum CommitmentAccumulator<F, PCS: StreamingCommitment<Field = F>> {
    Empty,
    Dense(DenseStreamingState<F, PCS>),
    OneHot(Vec<Option<usize>>),
}

impl<F, PCS> CommitmentAccumulator<F, PCS>
where
    F: jolt_field::Field,
    PCS: StreamingCommitment<Field = F>,
{
    const fn new() -> Self {
        Self::Empty
    }

    fn append<N: WitnessNamespace>(
        &mut self,
        chunk: PolynomialChunk<F>,
        row_width: usize,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match chunk {
            PolynomialChunk::Dense(values) => self.append_dense::<N>(values, row_width, setup),
            PolynomialChunk::U8(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u8).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::U16(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u16).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::U32(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u32).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::U64(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_u64).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::I64(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_i64).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::I128(values) => self.append_dense::<N>(
                values.into_iter().map(F::from_i128).collect(),
                row_width,
                setup,
            ),
            PolynomialChunk::OneHot(values) => self.append_one_hot::<N>(values),
        }
    }

    fn finish<N: WitnessNamespace>(
        self,
        polynomial_rows: usize,
        expected_pcs_rows: usize,
        setup: &PCS::ProverSetup,
    ) -> Result<(PCS::Output, PCS::OpeningHint), BackendError> {
        match self {
            Self::Empty => Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: "cannot commit an empty witness stream".to_owned(),
            }
            .into()),
            Self::Dense(state) => state.finish::<N>(expected_pcs_rows, setup),
            Self::OneHot(indices) => commit_one_hot::<PCS, N>(indices, polynomial_rows, setup),
        }
    }

    fn append_dense<N: WitnessNamespace>(
        &mut self,
        values: Vec<F>,
        row_width: usize,
        setup: &PCS::ProverSetup,
    ) -> Result<(), BackendError> {
        match self {
            Self::Dense(existing) => existing.append_values(values, setup),
            Self::OneHot(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "dense chunk followed one-hot chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => {
                let mut state = DenseStreamingState::<F, PCS>::new(row_width, setup);
                state.append_values(values, setup);
                *self = Self::Dense(state);
            }
        }
        Ok(())
    }

    fn append_one_hot<N: WitnessNamespace>(
        &mut self,
        values: Vec<Option<usize>>,
    ) -> Result<(), BackendError> {
        match self {
            Self::OneHot(existing) => existing.extend(values),
            Self::Dense(_) => {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "one-hot chunk followed dense chunk".to_owned(),
                }
                .into());
            }
            Self::Empty => *self = Self::OneHot(values),
        }
        Ok(())
    }
}

struct DenseStreamingState<F, PCS: StreamingCommitment<Field = F>> {
    partial: PCS::PartialCommitment,
    row_buffer: Vec<F>,
    rows_fed: usize,
    row_width: usize,
}

impl<F, PCS> DenseStreamingState<F, PCS>
where
    F: jolt_field::Field,
    PCS: StreamingCommitment<Field = F>,
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
        for value in values {
            self.row_buffer.push(value);
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
        Ok(PCS::finish_with_hint(self.partial, setup))
    }
}

pub(super) fn commit_streamed_witness<F, PCS, N>(
    slot: CommitmentSlot,
    oracle: jolt_witness::OracleRef<N>,
    polynomial_rows: usize,
    stream: &mut dyn PolynomialStream<F>,
    setup: &PCS::ProverSetup,
) -> Result<CpuCommitmentResult<N, PCS>, BackendError>
where
    F: jolt_field::Field,
    N: WitnessNamespace,
    PCS: StreamingCommitment<Field = F>,
{
    let mut chunks = Vec::new();
    let row_width = pcs_row_width::<N>(polynomial_rows)?;
    let expected_pcs_rows = polynomial_rows / row_width;
    let mut accumulator = CommitmentAccumulator::<F, PCS>::new();

    while let Some(chunk) = stream.next_chunk()? {
        let kind = chunk.kind();
        let index = chunks.len();
        chunks.push(StreamedWitnessChunk::new(index, kind, chunk.len()));
        accumulator.append::<N>(chunk, row_width, setup)?;
    }

    let streamed = StreamedWitnessOutput::new(slot, chunks);
    let (commitment, opening_hint) =
        accumulator.finish::<N>(polynomial_rows, expected_pcs_rows, setup)?;

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

fn commit_one_hot<PCS, N: WitnessNamespace>(
    indices: Vec<Option<usize>>,
    polynomial_rows: usize,
    setup: &PCS::ProverSetup,
) -> Result<(PCS::Output, PCS::OpeningHint), BackendError>
where
    PCS: StreamingCommitment,
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
    let indices = indices
        .into_iter()
        .map(|index| {
            index
                .map(|value| {
                    if value >= k || value > u8::MAX as usize {
                        Err(WitnessError::InvalidWitnessData {
                            namespace: N::ID.name,
                            reason: format!("one-hot column {value} is outside k={k} or u8 range",),
                        })
                    } else {
                        Ok(value as u8)
                    }
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let poly = jolt_poly::OneHotPolynomial::new(k, indices);
    Ok(PCS::commit(&poly, setup))
}
