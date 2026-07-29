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

use jolt_field::{Field, RingAccumulator};
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::JoltBackend;

pub mod booleanity;
pub mod bytecode_read_raf;
pub mod hamming_weight_claim_reduction;
pub mod inc_claim_reduction;
pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
pub mod opening;
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

/// The optimized implementations' marker type: implements the RAM-family
/// [`PrepareKernel`](crate::PrepareKernel) slots (each module here hosts its
/// impl next to the kernel it wraps).
pub struct OptimizedBackend;

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The optimized backend: [`JoltBackend::reference`] with every slot this
    /// module tree ports overwritten by its optimized kernel, so the two
    /// backends cannot drift on the slots left untouched (the precommitted /
    /// advice reduction slots, the commit slot, and the advice-opening
    /// evaluation). Same construction bounds as the reference backend (the
    /// commit slot is the reference streaming implementation), plus the
    /// accumulator bound the registers and shift kernels' compact-scalar
    /// walks need.
    pub fn optimized() -> Self
    where
        PCS: StreamingCommitment,
        F::Accumulator: RingAccumulator,
    {
        let mut backend = Self::reference();

        backend.spartan_outer_uniskip = Box::new(spartan_outer::OptimizedOuterUniskip);
        backend.spartan_outer_remainder = Box::new(spartan_outer::OptimizedOuterRemainder);
        backend.spartan_product_uniskip = Box::new(spartan_product::OptimizedProductUniskip);
        backend.spartan_product_remainder = Box::new(spartan_product::OptimizedProductRemainder);
        backend.spartan_shift = Box::new(spartan_shift::OptimizedSpartanShift);

        backend.ram_read_write = Box::new(OptimizedBackend);
        backend.ram_val_check = Box::new(OptimizedBackend);
        backend.ram_ra_claim_reduction = Box::new(OptimizedBackend);
        backend.ram_raf_evaluation = Box::new(OptimizedBackend);
        backend.ram_output_check = Box::new(OptimizedBackend);
        backend.ram_ra_virtualization = Box::new(OptimizedBackend);

        backend.instruction_read_raf = Box::new(instruction_read_raf::OptimizedInstructionReadRaf);
        backend.instruction_ra_virtualization =
            Box::new(instruction_ra_virtualization::OptimizedInstructionRaVirtualization);
        backend.instruction_claim_reduction =
            Box::new(instruction_claim_reduction::OptimizedInstructionClaimReduction);
        backend.instruction_input = Box::new(instruction_input::OptimizedInstructionInput);

        backend.registers_read_write = Box::new(registers_read_write::OptimizedRegistersReadWrite);
        backend.registers_val_evaluation =
            Box::new(registers_val_evaluation::OptimizedRegistersValEvaluation);
        backend.registers_claim_reduction =
            Box::new(registers_claim_reduction::OptimizedRegistersClaimReduction);

        backend.booleanity_address = Box::new(booleanity::OptimizedBooleanityAddress);
        backend.booleanity_cycle = Box::new(booleanity::OptimizedBooleanityCycle);
        backend.ram_hamming_booleanity =
            Box::new(ram_hamming_booleanity::OptimizedRamHammingBooleanity);

        backend.bytecode_read_raf_address = Box::new(OptimizedBytecodeReadRafAddress);
        backend.bytecode_read_raf_cycle = Box::new(OptimizedBytecodeReadRafCycle);
        backend.hamming_weight_claim_reduction = Box::new(OptimizedHammingWeightClaimReduction);
        backend.inc_claim_reduction = Box::new(OptimizedIncClaimReduction);

        backend.joint_opening = Box::new(OptimizedBackend);

        backend
    }
}

#[cfg(test)]
pub(crate) mod harness;
#[cfg(test)]
pub(crate) mod testing;
