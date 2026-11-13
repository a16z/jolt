//! Streaming kernel families: optional alternatives to the default eager
//! kernels. A streaming kernel recomputes its early rounds from typed rows
//! in task-partitioned passes instead of materializing trace-sized tables
//! immediately. [`JoltBackend::optimized`] keeps the eager kernels;
//! [`JoltBackend::with_streaming`] installs a selection.

mod lazy_ra;
mod pass;

pub(crate) use lazy_ra::{RowIndexSource, StreamedFoldedRa, MATERIALIZE_BYTES};
pub(crate) use pass::{
    bind_groups_in_place, dense_crossover, exact_task_fill, gruen_pair_pass, key_runs,
    product_endpoint_terms, task_ranges, FillMismatch, PairHalves, TaskSlots, DENSE_TABLE_CAP,
    VISIT_ROWS,
};

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use super::instruction_input::StreamingInstructionInput;
use super::instruction_ra_virtualization::StreamingInstructionRaVirtualization;
use super::ram_ra_claim_reduction::StreamingRamRaClaimReduction;
use super::ram_ra_virtualization::StreamingRamRaVirtualization;
use super::ram_raf_evaluation::StreamingRamRafEvaluation;
use super::ram_read_write::StreamingRamReadWrite;
use super::ram_val_check::StreamingRamValCheck;
use super::spartan_outer::StreamingOuterRemainder;
use super::spartan_product::StreamingProductRemainder;
use crate::JoltBackend;

/// The streaming kernel families a backend installs in place of its default
/// kernels ([`JoltBackend::with_streaming`]). A flag switches every slot of
/// its family together with the cross-stage state those slots share, so every
/// value is a valid selection; the default keeps every default kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFamilies {
    /// The Spartan outer (stage 1) and product (stage 2) remainders; their
    /// uni-skip carries are the default kernels'.
    pub spartan: bool,
    /// Stage 3's instruction input virtualization (no cross-stage state).
    pub instruction_input: bool,
    /// The RAM read-write and RAF checks (stage 2), value check (stage 4),
    /// RA claim reduction (stage 5) and RA virtualization (stage 6b): the
    /// RAM event log replaces the default kernels' address column, held from
    /// stage 2 through stage 5, and stage 6b collects its own column.
    pub ram: bool,
    /// Readers of the per-cycle instruction rows, which the default kernels
    /// share from stage 5 on as a packed row cache: stage 6b's instruction RA
    /// virtualization reads the witness's resident rows in its own passes.
    /// The cache's other readers keep their default kernels, so stage 5 still
    /// collects it. Stage 7's Hamming-weight claim reduction has no streaming
    /// kernel: its default kernel reads the resident rows whenever no cache
    /// is live.
    pub instruction_rows: bool,
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
            ram,
            instruction_rows,
        } = families;
        if spartan {
            self.spartan_outer_remainder = Box::new(StreamingOuterRemainder);
            self.spartan_product_remainder = Box::new(StreamingProductRemainder);
        }
        if instruction_input {
            self.instruction_input = Box::new(StreamingInstructionInput);
        }
        if ram {
            self.ram_read_write = Box::new(StreamingRamReadWrite);
            self.ram_raf_evaluation = Box::new(StreamingRamRafEvaluation);
            self.ram_val_check = Box::new(StreamingRamValCheck);
            self.ram_ra_claim_reduction = Box::new(StreamingRamRaClaimReduction);
            self.ram_ra_virtualization = Box::new(StreamingRamRaVirtualization);
        }
        if instruction_rows {
            self.instruction_ra_virtualization = Box::new(StreamingInstructionRaVirtualization);
        }
        self
    }
}
