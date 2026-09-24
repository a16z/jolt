use common::jolt_device::JoltDevice;

use super::{MemoryImage, TraceError, TraceInputs, TraceOutput, TraceRow};

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

/// Two-pass chunked execution: a fast checkpointing pass over the whole
/// program, then parallel per-chunk replay.
///
/// This is the producer-side contract for streaming consumers. Proof adapters
/// that require retained random access must emit their compact row format at
/// this boundary instead of draining a replaying source into another full
/// trace allocation.
pub trait ChunkedExecutionBackend: ExecutionBackend {
    /// Everything needed to deterministically re-execute one chunk,
    /// independent of every other chunk.
    type Checkpoint: Send + Sync;

    /// Fast pass: run the program to completion WITHOUT materializing trace
    /// rows. `checkpoints[i]` resumes at trace cycle `i * chunk_size`;
    /// `checkpoints.len() == trace_len.div_ceil(chunk_size)`.
    ///
    /// `max_trace_length` is not enforced here (parity with eager tracing,
    /// where enforcement happens at prove time); execution runs to guest
    /// termination.
    fn execute(
        &mut self,
        program: &super::JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
    ) -> Result<ExecutionSummary<Self::Checkpoint>, TraceError>;

    /// Recording pass: re-execute one chunk, materializing exactly
    /// `chunk_size` trace rows (fewer for the final chunk — replay emits no
    /// padding rows; padding stays the consumer's job, as in the existing
    /// `RowSource` contract). Takes `&self` so disjoint chunks can be
    /// replayed in parallel, in any order.
    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<Self::Trace, TraceError>;
}

/// Result of a [`ChunkedExecutionBackend::execute`] fast pass.
pub struct ExecutionSummary<C> {
    pub checkpoints: Vec<C>,
    pub trace_len: usize,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
    /// The populated runtime advice tape captured at guest termination
    /// (`None` when the backend produced no tape).
    pub advice_tape: Option<Vec<u8>>,
}
