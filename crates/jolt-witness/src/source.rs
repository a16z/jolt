//! Trace source abstraction.
//!
//! [`TraceSource`] defines the interface between execution trace backends
//! and witness generation. Implementing this trait allows any trace provider
//! (RISC-V emulator, hardware trace, alternative ISA) to plug into the Jolt
//! witness generation pipeline without changes to the proving system.

/// An execution trace that provides rows for witness generation.
///
/// Each row represents one cycle of execution. The witness builder iterates
/// over rows and extracts polynomial evaluation table entries.
///
/// # Type Parameter
///
/// The associated `Row` type is backend-specific. For the RISC-V tracer,
/// this is `tracer::Cycle`. For a hardware trace, it might be a different
/// struct capturing the same semantic information (register reads/writes,
/// memory operations, instruction identity).
///
/// # Contract
///
/// - [`len()`](Self::len) must return the exact number of rows.
/// - [`rows()`](Self::rows) must yield exactly `len()` items.
/// - Rows are consumed in order; random access is not required.
pub trait TraceSource {
    /// The type of a single trace row (one execution cycle).
    type Row;

    /// Total number of rows in the trace.
    ///
    /// Must be known upfront for table pre-allocation. The proving pipeline
    /// pads this to the next power of two.
    fn len(&self) -> usize;

    /// Whether the trace is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over trace rows in execution order.
    ///
    /// Called once during witness generation. Implementations may produce
    /// rows lazily (streaming from disk) or eagerly (in-memory `Vec`).
    fn rows(&self) -> impl ExactSizeIterator<Item = &Self::Row>;
}

/// Blanket implementation for `Vec<R>`: any vector of rows is a trace source.
impl<R> TraceSource for Vec<R> {
    type Row = R;

    fn len(&self) -> usize {
        <[R]>::len(self)
    }

    fn rows(&self) -> impl ExactSizeIterator<Item = &R> {
        self.iter()
    }
}

/// Blanket implementation for slices behind a reference.
impl<R> TraceSource for &[R] {
    type Row = R;

    fn len(&self) -> usize {
        <[R]>::len(self)
    }

    fn rows(&self) -> impl ExactSizeIterator<Item = &R> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct MockRow {
        pc: u64,
        rd_delta: i64,
    }

    #[test]
    fn vec_as_trace_source() {
        let trace = vec![
            MockRow { pc: 0, rd_delta: 1 },
            MockRow {
                pc: 4,
                rd_delta: -3,
            },
            MockRow { pc: 8, rd_delta: 0 },
        ];

        assert_eq!(TraceSource::len(&trace), 3);
        assert!(!trace.is_empty());

        let pcs: Vec<u64> = trace.rows().map(|r| r.pc).collect();
        assert_eq!(pcs, vec![0, 4, 8]);
    }

    #[test]
    fn slice_as_trace_source() {
        let data = [
            MockRow { pc: 0, rd_delta: 1 },
            MockRow { pc: 4, rd_delta: 2 },
        ];
        let trace: &[MockRow] = &data;

        assert_eq!(TraceSource::len(&trace), 2);
        let deltas: Vec<i64> = trace.rows().map(|r| r.rd_delta).collect();
        assert_eq!(deltas, vec![1, 2]);
    }

    #[test]
    fn empty_trace() {
        let trace: Vec<MockRow> = vec![];
        assert_eq!(TraceSource::len(&trace), 0);
        assert!(trace.is_empty());
        assert_eq!(trace.rows().count(), 0);
    }
}
