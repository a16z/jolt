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
    product_endpoint_terms, split_at_tasks, task_ranges, FillMismatch, GruenSums, PairHalves,
    TaskSlots, DENSE_TABLE_CAP, VISIT_ROWS,
};

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use super::booleanity::{StreamingBooleanityAddress, StreamingBooleanityCycle};
use super::instruction_input::StreamingInstructionInput;
use super::instruction_ra_virtualization::StreamingInstructionRaVirtualization;
use super::instruction_read_raf::StreamingInstructionReadRaf;
use super::ram_ra_claim_reduction::StreamingRamRaClaimReduction;
use super::ram_ra_virtualization::StreamingRamRaVirtualization;
use super::ram_raf_evaluation::StreamingRamRafEvaluation;
use super::ram_read_write::StreamingRamReadWrite;
use super::ram_val_check::StreamingRamValCheck;
use super::registers_read_write::StreamingRegistersReadWrite;
use super::registers_val_evaluation::StreamingRegistersValEvaluation;
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
    /// The register read-write check (stage 4) and value evaluation
    /// (stage 5): both read the rows, so no `rd` index column carries from
    /// stage 4 to stage 5, and under cycle-first binding stage 4 keeps no
    /// per-cycle columns. An address-first stage 4 keeps the default kernel.
    pub registers: bool,
    /// Readers of the per-cycle instruction rows, which the default kernels
    /// share from stage 5 on as a packed row cache: the instruction read-RAF
    /// check (stage 5), the booleanity address phase (stage 6a) and stage
    /// 6b's booleanity cycle phase and instruction RA virtualization read the
    /// witness's resident rows in their own passes, which stop at the rows'
    /// constant suffix as the default kernels' tables do. The bytecode
    /// read-RAF kernels keep their default kernels, so the cache is no longer
    /// collected in stage 5 but by stage 6a's bytecode read-RAF address
    /// phase, and lives until the stage-6b bytecode read-RAF cycle kernel
    /// releases it. Stage 7's Hamming-weight claim reduction has no streaming
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
            registers,
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
        if registers {
            self.registers_read_write = Box::new(StreamingRegistersReadWrite);
            self.registers_val_evaluation = Box::new(StreamingRegistersValEvaluation);
        }
        if instruction_rows {
            self.instruction_read_raf = Box::new(StreamingInstructionReadRaf);
            self.booleanity_address = Box::new(StreamingBooleanityAddress);
            self.instruction_ra_virtualization = Box::new(StreamingInstructionRaVirtualization);
            self.booleanity_cycle = Box::new(StreamingBooleanityCycle);
        }
        self
    }
}
