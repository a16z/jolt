//! The optimized backend: legacy-ported RAM-family kernels behind the same
//! slots the reference backend serves, byte-identical round polynomials and
//! output claims by construction (the parity tests in each module pin them
//! against the naive tier on synthetic traces; `byte_diff` pins the full
//! proofs against `jolt-prover-legacy`).
//!
//! Wave 1 covers the four RAM relations whose reference kernels materialize
//! the dense `(K × T)` `ra`/`val` grids:
//!
//! - [`ram_read_write`]: the full legacy port — phased sparse read-write
//!   matrix (cycle-major then address-major) with Gruen split-eq cycle
//!   rounds. `O(accesses)` state instead of `O(K·T)`.
//! - [`ram_val_check`], [`ram_ra_claim_reduction`], [`ram_raf_evaluation`]:
//!   their round loops were already over single-dimension tables; only the
//!   reference `prepare` exploded (grid materialization + fold). These
//!   kernels build the *same* tables in `O(T + K)` from one shared trace
//!   walk and hand them to the naive round loop, so parity is structural.
//!
//! Every other slot stays on the reference tier.

use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::reference::precommitted_reduction::ReferencePrecommittedAddress;
use crate::reference::spartan_outer::ReferenceOuterRemainder;
use crate::reference::spartan_product::ReferenceProductRemainder;
use crate::{JoltBackend, ReferenceBackend};

pub mod booleanity;
pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
pub mod ram_hamming_booleanity;
pub mod ram_ra_claim_reduction;
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

/// The optimized implementations' marker type: implements the RAM-family
/// [`PrepareKernel`](crate::PrepareKernel) slots (each module here hosts its
/// impl next to the kernel it wraps).
pub struct OptimizedBackend;

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The optimized backend: the four RAM-family slots served by the
    /// legacy-ported kernels in this module, every other slot identical to
    /// [`JoltBackend::reference`]. Same construction bounds as the reference
    /// backend (the commit slot is the reference streaming implementation).
    pub fn optimized() -> Self
    where
        PCS: StreamingCommitment,
    {
        Self {
            commit: Box::new(ReferenceBackend),
            spartan_outer_uniskip: Box::new(ReferenceBackend),
            spartan_outer_remainder: Box::new(ReferenceOuterRemainder),
            spartan_product_uniskip: Box::new(ReferenceBackend),
            spartan_product_remainder: Box::new(ReferenceProductRemainder),
            ram_read_write: Box::new(OptimizedBackend),
            instruction_claim_reduction: Box::new(ReferenceBackend),
            ram_raf_evaluation: Box::new(OptimizedBackend),
            ram_output_check: Box::new(ReferenceBackend),
            spartan_shift: Box::new(ReferenceBackend),
            instruction_input: Box::new(ReferenceBackend),
            registers_claim_reduction: Box::new(ReferenceBackend),
            registers_read_write: Box::new(ReferenceBackend),
            ram_val_check: Box::new(OptimizedBackend),
            advice_opening: Box::new(ReferenceBackend),
            instruction_read_raf: Box::new(ReferenceBackend),
            ram_ra_claim_reduction: Box::new(OptimizedBackend),
            registers_val_evaluation: Box::new(ReferenceBackend),
            bytecode_read_raf_address: Box::new(ReferenceBackend),
            booleanity_address: Box::new(ReferenceBackend),
            bytecode_read_raf_cycle: Box::new(ReferenceBackend),
            booleanity_cycle: Box::new(ReferenceBackend),
            ram_hamming_booleanity: Box::new(ReferenceBackend),
            ram_ra_virtualization: Box::new(ReferenceBackend),
            instruction_ra_virtualization: Box::new(ReferenceBackend),
            inc_claim_reduction: Box::new(ReferenceBackend),
            trusted_advice_cycle: Box::new(ReferenceBackend),
            untrusted_advice_cycle: Box::new(ReferenceBackend),
            bytecode_reduction_cycle: Box::new(ReferenceBackend),
            program_image_reduction_cycle: Box::new(ReferenceBackend),
            hamming_weight_claim_reduction: Box::new(ReferenceBackend),
            trusted_advice_address: Box::new(ReferencePrecommittedAddress::new(
                "stage 6b parked no trusted-advice reduction state for the scheduled address phase",
            )),
            untrusted_advice_address: Box::new(ReferencePrecommittedAddress::new(
                "stage 6b parked no untrusted-advice reduction state for the scheduled address phase",
            )),
            bytecode_reduction_address: Box::new(ReferencePrecommittedAddress::new(
                "stage 6b parked no bytecode reduction state for the scheduled address phase",
            )),
            program_image_reduction_address: Box::new(ReferencePrecommittedAddress::new(
                "stage 6b parked no program-image reduction state for the scheduled address phase",
            )),
            joint_opening: Box::new(ReferenceBackend),
        }
    }
}

#[cfg(test)]
pub(crate) mod testing;
