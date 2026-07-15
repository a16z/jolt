use crate::WitnessError;

/// One chunk of a committed polynomial's coefficient stream, borrowed from
/// the walker's buffer. Variants are the scalar encodings backends actually
/// produce; consumers promote to field elements (or feed a commitment
/// scheme's typed fast paths) at their boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommittedChunk<'a, F> {
    /// Field-element coefficients (stored columns, field-inline values).
    Dense(&'a [F]),
    /// A run of zero coefficients.
    Zeros(usize),
    /// Little-endian `u64` words (advice bytes).
    Words(&'a [u64]),
    /// Signed increments.
    Increments(&'a [i128]),
    /// Per-cycle hot addresses of a one-hot `(K x T)` grid column; `None` is
    /// a cold cycle.
    HotAddresses(&'a [Option<usize>]),
}

impl<F> CommittedChunk<'_, F> {
    pub const fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Zeros(rows) => *rows,
            Self::Words(values) => values.len(),
            Self::Increments(values) => values.len(),
            Self::HotAddresses(values) => values.len(),
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The per-chunk callback of a committed-column walk
/// ([`crate::JoltWitnessOracle::visit_committed_column`]).
pub type ColumnVisitor<'a, F> = dyn FnMut(CommittedChunk<'_, F>) -> Result<(), WitnessError> + 'a;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_report_logical_row_counts() {
        assert_eq!(CommittedChunk::<u64>::Increments(&[1, -2, 3]).len(), 3);
        assert_eq!(CommittedChunk::<u64>::Zeros(5).len(), 5);
        assert!(!CommittedChunk::<u64>::HotAddresses(&[Some(4), None]).is_empty());
    }
}
