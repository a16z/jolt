//! The reference backend: every [`JoltBackend`] slot served by the naive
//! tier, plus the hand-written kernels the naive interpreter cannot express
//! (post-uni-skip stream rounds, prefix–suffix decompositions, staged
//! intermediates hiding product-of-multilinear summands).
//!
//! Each module here fills its [`JoltBackend`] slot — a
//! [`PrepareKernel`](crate::PrepareKernel) impl per sumcheck relation,
//! resolved through [`HasKernel`](crate::HasKernel), or a bespoke slot trait
//! impl; the seam (the traits, [`JoltBackend`], `ProofSession`, the wire
//! types) never depends on this directory. The reference tier is the
//! equivalence anchor optimized backends are tested against and the
//! fallback partial backends compose over; it is eager-dense throughout — a
//! test oracle at harness scale, never a performance path.

use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::JoltBackend;

use self::precommitted_reduction::ReferencePrecommittedAddress;
use self::spartan_outer::ReferenceOuterRemainder;
use self::spartan_product::ReferenceProductRemainder;

pub mod advice_claim_reduction;
pub mod booleanity;
pub mod bytecode_claim_reduction;
pub mod bytecode_read_raf;
pub mod commitment;
pub mod hamming_weight_claim_reduction;
pub mod inc_claim_reduction;
pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
pub mod naive;
pub mod opening;
pub mod precommitted_reduction;
pub mod program_image_claim_reduction;
pub mod ram_hamming_booleanity;
pub mod ram_output_check;
pub mod ram_ra_claim_reduction;
pub mod ram_ra_virtualization;
pub mod ram_raf_evaluation;
pub mod ram_read_write;
pub mod ram_val_check;
pub mod registers_claim_reduction;
pub mod registers_read_write;
pub mod registers_val_evaluation;
pub mod spartan_outer;
pub mod spartan_product;
pub mod spartan_shift;
pub(crate) mod views;

/// The reference implementations' marker type: implements every slot trait
/// (each module here hosts its impl next to the kernel it wraps).
pub struct ReferenceBackend;

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The always-present reference backend: every slot served by the naive
    /// tier. It is the equivalence anchor optimized backends are tested
    /// against, and the fallback partial backends compose over. Its commit
    /// slot streams, hence the [`StreamingCommitment`] bound — a
    /// reference-implementation requirement, not a seam one.
    pub fn reference() -> Self
    where
        PCS: StreamingCommitment,
    {
        Self {
            commit: Box::new(ReferenceBackend),
            spartan_outer_uniskip: Box::new(ReferenceBackend),
            spartan_outer_remainder: Box::new(ReferenceOuterRemainder),
            spartan_product_uniskip: Box::new(ReferenceBackend),
            spartan_product_remainder: Box::new(ReferenceProductRemainder),
            ram_read_write: Box::new(ReferenceBackend),
            instruction_claim_reduction: Box::new(ReferenceBackend),
            ram_raf_evaluation: Box::new(ReferenceBackend),
            ram_output_check: Box::new(ReferenceBackend),
            spartan_shift: Box::new(ReferenceBackend),
            instruction_input: Box::new(ReferenceBackend),
            registers_claim_reduction: Box::new(ReferenceBackend),
            registers_read_write: Box::new(ReferenceBackend),
            ram_val_check: Box::new(ReferenceBackend),
            advice_opening: Box::new(ReferenceBackend),
            instruction_read_raf: Box::new(ReferenceBackend),
            ram_ra_claim_reduction: Box::new(ReferenceBackend),
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
