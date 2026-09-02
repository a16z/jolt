//! The optimized backend: legacy-ported kernels behind the same slots the
//! reference backend serves, byte-identical round polynomials and output
//! claims by construction (the parity tests in each module pin them against
//! the naive tier on synthetic traces; `byte_diff` pins the full proofs
//! against `jolt-prover-legacy`).
//!
//! The shared playbook, per kernel:
//! - **Sparse one-hot access**: per-cycle hot indices off typed witness
//!   bundles ([`jolt_witness::collect_bundles`]) replace `oracle_table` walks
//!   over materialized `K x T` one-hot grids — `O(T)` per polynomial instead
//!   of `O(K·T)`.
//! - **Split-eq pushforwards**: `Σ_{j: idx(j)=k} eq(r, j)` accumulated as
//!   `E_hi[j_hi] · (Σ_{j_lo} E_lo[j_lo])` — inner sums are additions only,
//!   one multiplication per touched slot per outer block.
//! - **Linear-leaf fusion**: eq/selector leaves that enter the summand
//!   linearly are folded into one combined coefficient table (or a constant),
//!   shrinking per-round bind and extension work; exactness of multilinear
//!   extension under scalar-weighted sums keeps the round messages
//!   byte-identical.
//! - **Eval-at-1 recovery**: round messages sample the summand at
//!   `t ∈ {0, 2, .., degree}` and recover `s(1) = previous_claim − s(0)`,
//!   the same trade the legacy prover makes (a dishonest input claim
//!   surfaces at the driver's final-claim check instead of the round check).
//! - **Rayon cycle walks** with per-thread partial accumulators.
//!
//! Each module documents its own port; [`JoltBackend::optimized`] wires them.

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use crate::commitment::ModeStreamingCommitment;

use crate::JoltBackend;

pub mod booleanity;
pub mod bytecode_read_raf;
pub mod commitment;
pub mod hamming_weight_claim_reduction;
pub mod inc_claim_reduction;
pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
mod lazy_ra;
pub mod opening;
pub mod precommitted_reduction;
pub mod ram_hamming_booleanity;
pub mod ram_output_check;
pub mod ram_ra_claim_reduction;
pub mod ram_ra_virtualization;
pub mod ram_raf_evaluation;
pub mod ram_read_write;
mod ram_trace;
pub mod ram_val_check;
pub mod registers_claim_reduction;
pub mod registers_read_write;
pub mod registers_val_evaluation;
mod rw_matrix;
pub mod spartan_outer;
pub mod spartan_product;
pub mod spartan_shift;
mod support;

pub use bytecode_read_raf::{OptimizedBytecodeReadRafAddress, OptimizedBytecodeReadRafCycle};
pub use hamming_weight_claim_reduction::OptimizedHammingWeightClaimReduction;
pub use inc_claim_reduction::OptimizedIncClaimReduction;
pub use precommitted_reduction::{OptimizedPrecommittedAddress, OptimizedPrecommittedCycle};

/// The optimized implementations' marker type: implements the RAM-family
/// [`PrepareKernel`](crate::PrepareKernel) slots (each module here hosts its
/// impl next to the kernel it wraps).
pub struct OptimizedBackend;

impl<F, PCS> JoltBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// The optimized backend: [`JoltBackend::reference`] with every slot this
    /// module tree ports overwritten by its optimized kernel. Same
    /// construction bounds as the reference backend.
    pub fn optimized() -> Self
    where
        PCS: ModeStreamingCommitment,
    {
        let mut backend = Self::reference().with_optimized_compute();

        backend.commit = Box::new(OptimizedBackend);
        backend.joint_opening = Box::new(OptimizedBackend);

        backend
    }

    /// Replace every protocol-arithmetic slot with its optimized kernel while
    /// preserving the commitment and opening slots owned by the caller.
    ///
    /// Packed commitment schemes use native group commitment/opening paths, so
    /// they cannot satisfy [`jolt_openings::StreamingCommitment`] and must retain their own
    /// boundary slots while sharing the optimized stage 1–7 kernels.
    pub fn with_optimized_compute(mut self) -> Self {
        self.spartan_outer_uniskip = Box::new(spartan_outer::OptimizedOuterUniskip);
        self.spartan_outer_remainder = Box::new(spartan_outer::OptimizedOuterRemainder);
        self.spartan_product_uniskip = Box::new(spartan_product::OptimizedProductUniskip);
        self.spartan_product_remainder = Box::new(spartan_product::OptimizedProductRemainder);
        self.spartan_shift = Box::new(spartan_shift::OptimizedSpartanShift);

        self.ram_read_write = Box::new(OptimizedBackend);
        self.ram_val_check = Box::new(OptimizedBackend);
        self.ram_ra_claim_reduction = Box::new(OptimizedBackend);
        self.ram_raf_evaluation = Box::new(OptimizedBackend);
        self.ram_output_check = Box::new(OptimizedBackend);
        self.ram_ra_virtualization = Box::new(OptimizedBackend);

        self.instruction_read_raf = Box::new(instruction_read_raf::OptimizedInstructionReadRaf);
        self.instruction_ra_virtualization =
            Box::new(instruction_ra_virtualization::OptimizedInstructionRaVirtualization);
        self.instruction_claim_reduction =
            Box::new(instruction_claim_reduction::OptimizedInstructionClaimReduction);
        self.instruction_input = Box::new(instruction_input::OptimizedInstructionInput);

        self.registers_read_write = Box::new(registers_read_write::OptimizedRegistersReadWrite);
        self.registers_val_evaluation =
            Box::new(registers_val_evaluation::OptimizedRegistersValEvaluation);
        self.registers_claim_reduction =
            Box::new(registers_claim_reduction::OptimizedRegistersClaimReduction);

        self.booleanity_address = Box::new(booleanity::OptimizedBooleanityAddress);
        self.booleanity_cycle = Box::new(booleanity::OptimizedBooleanityCycle);
        self.ram_hamming_booleanity =
            Box::new(ram_hamming_booleanity::OptimizedRamHammingBooleanity);

        self.bytecode_read_raf_address = Box::new(OptimizedBytecodeReadRafAddress);
        self.bytecode_read_raf_cycle = Box::new(OptimizedBytecodeReadRafCycle);
        self.hamming_weight_claim_reduction = Box::new(OptimizedHammingWeightClaimReduction);
        self.inc_claim_reduction = Box::new(OptimizedIncClaimReduction);

        self.trusted_advice_cycle = Box::new(OptimizedPrecommittedCycle);
        self.untrusted_advice_cycle = Box::new(OptimizedPrecommittedCycle);
        self.bytecode_reduction_cycle = Box::new(OptimizedPrecommittedCycle);
        self.program_image_reduction_cycle = Box::new(OptimizedPrecommittedCycle);
        self.advice_opening = Box::new(OptimizedPrecommittedCycle);
        self.trusted_advice_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no trusted-advice reduction state for the scheduled address phase",
        ));
        self.untrusted_advice_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no untrusted-advice reduction state for the scheduled address phase",
        ));
        self.bytecode_reduction_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no bytecode reduction state for the scheduled address phase",
        ));
        self.program_image_reduction_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no program-image reduction state for the scheduled address phase",
        ));

        self
    }
}

#[cfg(test)]
pub(crate) mod parity;
#[cfg(test)]
pub(crate) mod testing;
