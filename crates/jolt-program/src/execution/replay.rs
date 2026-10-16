use jolt_riscv::JoltTraceRow;

use super::TraceError;

/// The rows of one recorded execution, re-executed on demand instead of
/// retained. The witness reads rows through this trait so it does not depend
/// on an emulator; the tracer's replay producer
/// (`TracerBackend::execute_replay`) implements it.
pub trait RowReplay: Send + Sync {
    /// Checkpoint spacing in rows, a power of two. A reader whose first row is
    /// a multiple of it starts from that row's checkpoint and re-executes no
    /// earlier rows, unless one multi-row instruction spans the multiple.
    fn segment_rows(&self) -> usize;

    /// Rows before canonical trailing padding: the proof prefix that
    /// [`TraceData::proof_len`](super::TraceData::proof_len) reports for the
    /// same execution. Padding rows are not replayed.
    fn proof_len(&self) -> usize;

    /// A forward reader whose first row is `start`, which must be below
    /// [`Self::proof_len`]. Readers own their emulator state, share nothing
    /// with each other, and run on the calling thread.
    fn reader(&self, start: usize) -> Result<Box<dyn RowReader + '_>, TraceError>;
}

/// Sequential reader over a [`RowReplay`].
pub trait RowReader {
    /// Writes the next `rows.len()` rows. A request reaching past the proof
    /// prefix fails before any row is written; after a replay error the
    /// contents of `rows` are unspecified.
    fn read(&mut self, rows: &mut [JoltTraceRow]) -> Result<(), TraceError>;
}
