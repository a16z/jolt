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
}
