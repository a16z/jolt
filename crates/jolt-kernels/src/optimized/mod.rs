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
//! The playbook's shared machinery lives in `support` (the split-eq round
//! driver `GruenRoundMessage`, `RoundProgress`, the `pin_derived_term`
//! drift checks, accumulator and parallel-fold helpers) and `lazy_ra` (the
//! lazy one-hot fold); `ram_trace` and `rw_matrix` carry the RAM family's
//! shared trace columns and sparse matrix. Each kernel module documents its
//! own port; [`JoltBackend::optimized`] wires them.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_witness::JoltWitnessPlane;

use crate::commitment::ModeStreamingCommitment;

use crate::JoltBackend;

use self::booleanity::{OptimizedBooleanityAddress, OptimizedBooleanityCycle};
use self::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use self::instruction_input::OptimizedInstructionInput;
use self::instruction_ra_virtualization::OptimizedInstructionRaVirtualization;
use self::instruction_read_raf::OptimizedInstructionReadRaf;
use self::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use self::registers_claim_reduction::OptimizedRegistersClaimReduction;
use self::registers_read_write::OptimizedRegistersReadWrite;
use self::registers_val_evaluation::OptimizedRegistersValEvaluation;
use self::spartan_outer::{OptimizedOuterRemainder, OptimizedOuterUniskip};
use self::spartan_product::{OptimizedProductRemainder, OptimizedProductUniskip};
use self::spartan_shift::OptimizedSpartanShift;

#[cfg(feature = "allocative")]
macro_rules! impl_field_allocative {
    ($type:ident, |$value:ident| $heap:block) => {
        impl<F: jolt_field::Field> allocative::Allocative for $type<F> {
            fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                let mut visitor = visitor.enter_self_sized::<Self>();
                let $value = self;
                let heap_bytes: usize = $heap;
                visitor.visit_simple(allocative::Key::new("heap"), heap_bytes);
                visitor.exit();
            }
        }
    };
}

#[cfg(feature = "allocative")]
pub(crate) use impl_field_allocative;

#[cfg(feature = "allocative")]
macro_rules! impl_allocative {
    ($type:ty, |$value:ident| $heap:block) => {
        impl allocative::Allocative for $type {
            fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                let mut visitor = visitor.enter_self_sized::<Self>();
                let $value = self;
                let heap_bytes: usize = $heap;
                visitor.visit_simple(allocative::Key::new("heap"), heap_bytes);
                visitor.exit();
            }
        }
    };
}

#[cfg(feature = "allocative")]
pub(crate) use impl_allocative;

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
pub(crate) mod rw_matrix;
pub mod spartan_outer;
pub mod spartan_product;
pub mod spartan_shift;
pub(crate) mod support;

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
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The optimized backend: [`JoltBackend::reference`] with every slot this
    /// module tree ports overwritten by its optimized kernel. Same
    /// construction bounds as the reference backend.
    pub fn optimized() -> Self
    where
        PCS: ModeStreamingCommitment,
    {
        let mut backend = Self::reference();

        backend.commit = Box::new(OptimizedBackend);

        backend.spartan_outer_uniskip = Box::new(OptimizedOuterUniskip);
        backend.spartan_outer_remainder = Box::new(OptimizedOuterRemainder);
        backend.spartan_product_uniskip = Box::new(OptimizedProductUniskip);
        backend.spartan_product_remainder = Box::new(OptimizedProductRemainder);
        backend.spartan_shift = Box::new(OptimizedSpartanShift);

        backend.ram_read_write = Box::new(OptimizedBackend);
        backend.ram_val_check = Box::new(OptimizedBackend);
        backend.ram_ra_claim_reduction = Box::new(OptimizedBackend);
        backend.ram_raf_evaluation = Box::new(OptimizedBackend);
        backend.ram_output_check = Box::new(OptimizedBackend);
        backend.ram_ra_virtualization = Box::new(OptimizedBackend);

        backend.instruction_read_raf = Box::new(OptimizedInstructionReadRaf);
        backend.instruction_ra_virtualization = Box::new(OptimizedInstructionRaVirtualization);
        backend.instruction_claim_reduction = Box::new(OptimizedInstructionClaimReduction);
        backend.instruction_input = Box::new(OptimizedInstructionInput);

        backend.registers_read_write = Box::new(OptimizedRegistersReadWrite);
        backend.registers_val_evaluation = Box::new(OptimizedRegistersValEvaluation);
        backend.registers_claim_reduction = Box::new(OptimizedRegistersClaimReduction);

        backend.booleanity_address = Box::new(OptimizedBooleanityAddress);
        backend.booleanity_cycle = Box::new(OptimizedBooleanityCycle);
        backend.ram_hamming_booleanity = Box::new(OptimizedRamHammingBooleanity);

        backend.bytecode_read_raf_address = Box::new(OptimizedBytecodeReadRafAddress);
        backend.bytecode_read_raf_cycle = Box::new(OptimizedBytecodeReadRafCycle);
        backend.hamming_weight_claim_reduction = Box::new(OptimizedHammingWeightClaimReduction);
        backend.inc_claim_reduction = Box::new(OptimizedIncClaimReduction);

        backend.joint_opening = Box::new(OptimizedBackend);

        backend.trusted_advice_cycle = Box::new(OptimizedPrecommittedCycle);
        backend.untrusted_advice_cycle = Box::new(OptimizedPrecommittedCycle);
        backend.bytecode_reduction_cycle = Box::new(OptimizedPrecommittedCycle);
        backend.program_image_reduction_cycle = Box::new(OptimizedPrecommittedCycle);
        backend.advice_opening = Box::new(OptimizedPrecommittedCycle);
        backend.trusted_advice_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no trusted-advice reduction state for the scheduled address phase",
        ));
        backend.untrusted_advice_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no untrusted-advice reduction state for the scheduled address phase",
        ));
        backend.bytecode_reduction_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no bytecode reduction state for the scheduled address phase",
        ));
        backend.program_image_reduction_address = Box::new(OptimizedPrecommittedAddress::new(
            "stage 6b parked no program-image reduction state for the scheduled address phase",
        ));

        backend
    }
}

#[cfg(test)]
pub(crate) mod parity;
pub(crate) mod rows;
#[cfg(test)]
pub(crate) mod testing;

pub fn warm_shared_witness<F: Field>(
    session: &mut crate::ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
) -> Result<(), crate::KernelError<F>> {
    let cycles = 1usize << log_t;
    let _ = self::ram_trace::RamAccessColumns::shared(session, witness, log_t)?;
    let _ = self::bytecode_read_raf::PcRow::shared(session, witness, cycles)?;
    let _ = self::instruction_read_raf::InstructionCycleRow::shared(session, witness, cycles)?;
    Ok(())
}
