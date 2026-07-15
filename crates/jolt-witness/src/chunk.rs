use crate::WitnessError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialChunkKind {
    Dense,
    Zeros,
    U8,
    U16,
    U32,
    U64,
    I64,
    I128,
    OneHot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolynomialChunk<F> {
    Dense(Vec<F>),
    Zeros(usize),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    I128(Vec<i128>),
    OneHot(Vec<Option<usize>>),
}

impl<F> PolynomialChunk<F> {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Zeros(rows) => *rows,
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::I128(values) => values.len(),
            Self::OneHot(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn kind(&self) -> PolynomialChunkKind {
        match self {
            Self::Dense(_) => PolynomialChunkKind::Dense,
            Self::Zeros(_) => PolynomialChunkKind::Zeros,
            Self::U8(_) => PolynomialChunkKind::U8,
            Self::U16(_) => PolynomialChunkKind::U16,
            Self::U32(_) => PolynomialChunkKind::U32,
            Self::U64(_) => PolynomialChunkKind::U64,
            Self::I64(_) => PolynomialChunkKind::I64,
            Self::I128(_) => PolynomialChunkKind::I128,
            Self::OneHot(_) => PolynomialChunkKind::OneHot,
        }
    }
}

pub trait PolynomialStream<F> {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<F>>, WitnessError>;
}

/// One lockstep chunk of a batched stream: the same row range of every
/// polynomial in the batch, keyed by the protocol's committed id type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialBatchChunk<Id, F> {
    pub chunks: Vec<(Id, PolynomialChunk<F>)>,
}

impl<Id, F> PolynomialBatchChunk<Id, F> {
    pub fn new(chunks: Vec<(Id, PolynomialChunk<F>)>) -> Self {
        Self { chunks }
    }

    pub fn len(&self) -> usize {
        self.chunks.first().map_or(0, |(_, chunk)| chunk.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait PolynomialBatchStream<F, Id> {
    fn next_batch(&mut self) -> Result<Option<PolynomialBatchChunk<Id, F>>, WitnessError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_chunks_report_logical_row_count() {
        let chunk = PolynomialChunk::<u64>::U8(vec![1, 2, 3]);

        assert_eq!(chunk.len(), 3);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn zero_chunks_report_logical_row_count() {
        let chunk = PolynomialChunk::<u64>::Zeros(5);

        assert_eq!(chunk.len(), 5);
        assert_eq!(chunk.kind(), PolynomialChunkKind::Zeros);
    }

    #[test]
    fn one_hot_chunks_preserve_sparse_row_positions() {
        let chunk = PolynomialChunk::<u64>::OneHot(vec![Some(4), None, Some(7)]);

        assert_eq!(chunk.len(), 3);
        assert_eq!(chunk.kind(), PolynomialChunkKind::OneHot);
    }
}
