//! Streaming kernel families: optional alternatives to the default eager
//! kernels for proofs whose eager tables would not fit memory. A streaming
//! kernel recomputes its early rounds from the witness's resident rows in
//! task-partitioned passes instead of materializing trace-sized tables, so it
//! trades prover time for peak memory. [`JoltBackend::optimized`] keeps the
//! eager kernels; [`JoltBackend::with_streaming`] installs a selection.

mod pass;

pub(crate) use pass::{
    gruen_pair_pass, pass_cycles, task_ranges, PairHalves, DENSE_TABLE_CAP, VISIT_ROWS,
};

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use super::instruction_input::StreamingInstructionInput;
use crate::JoltBackend;

/// The streaming kernel families a backend installs in place of its default
/// kernels ([`JoltBackend::with_streaming`]). A flag switches every slot of
/// its family together with the cross-stage state those slots share, so every
/// value is a valid selection; the default keeps every default kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFamilies {
    /// Stage 3's instruction input virtualization (no cross-stage state).
    pub instruction_input: bool,
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Replaces every slot of each family `families` selects with its
    /// streaming kernel; every other slot keeps its kernel. Called on
    /// [`JoltBackend::optimized`] before the first stage.
    pub fn with_streaming(mut self, families: StreamingFamilies) -> Self {
        let StreamingFamilies { instruction_input } = families;
        if instruction_input {
            self.instruction_input = Box::new(StreamingInstructionInput);
        }
        self
    }
}
