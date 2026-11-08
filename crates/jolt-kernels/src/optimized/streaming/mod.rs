//! Streaming kernel families: optional alternatives to the default eager
//! kernels. A streaming kernel recomputes its early rounds from typed rows
//! in task-partitioned passes instead of materializing trace-sized tables
//! immediately. [`JoltBackend::optimized`] keeps the eager kernels;
//! [`JoltBackend::with_streaming`] installs a selection.

mod pass;

pub(crate) use pass::{
    dense_crossover, gruen_pair_pass, product_endpoint_terms, task_ranges, PairHalves,
    DENSE_TABLE_CAP, VISIT_ROWS,
};

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use super::instruction_input::StreamingInstructionInput;
use super::spartan_outer::StreamingOuterRemainder;
use crate::JoltBackend;

/// The streaming kernel families a backend installs in place of its default
/// kernels ([`JoltBackend::with_streaming`]). A flag switches every slot of
/// its family together with the cross-stage state those slots share, so every
/// value is a valid selection; the default keeps every default kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFamilies {
    /// The Spartan outer remainder (stage 1); its uni-skip carry is the
    /// default kernel's.
    pub spartan: bool,
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
        let StreamingFamilies {
            spartan,
            instruction_input,
        } = families;
        if spartan {
            self.spartan_outer_remainder = Box::new(StreamingOuterRemainder);
        }
        if instruction_input {
            self.instruction_input = Box::new(StreamingInstructionInput);
        }
        self
    }
}
