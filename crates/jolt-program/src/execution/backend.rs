use super::{TraceError, TraceInputs, TraceOutput, TraceRow};

pub trait ExecutionBackend {
    type Trace: TraceSource;

    fn trace(
        &mut self,
        program: &super::JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError>;
}

pub trait TraceSource {
    fn next_row(&mut self) -> Option<TraceRow>;

    /// The full row sequence as one slice, if this source can serve it.
    ///
    /// Contract: the slice must equal exactly what the remaining `next_row`
    /// calls would yield — a partially consumed source must return `None`
    /// rather than a slice that includes already-consumed rows.
    fn rows(&self) -> Option<&[TraceRow]> {
        None
    }
}
