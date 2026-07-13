//! The shared precommitted claim-reduction member: the two-phase (stage-6b
//! cycle phase → stage-7 address phase) instance trait the advice,
//! committed-bytecode, and program-image reduction slots return.

use jolt_field::Field;
use jolt_sumcheck::ProveRounds;

use crate::KernelError;

/// One precommitted reduction member, spanning stages 6b and 7: the stage-6b
/// recipe drives the cycle phase and stages the handoff claim, then hands the
/// SAME object to stage 7, which calls
/// [`Self::transition_to_address_phase`] and drives the address phase (only
/// when the schedule has active address rounds).
pub trait PrecommittedReductionProver<F: Field>: ProveRounds<F> {
    /// Flip the member from the cycle phase to the address phase. Call between
    /// the stage-6b and stage-7 batches.
    fn transition_to_address_phase(&mut self);

    /// The intermediate claim staged at the cycle→address handoff:
    /// `Σ_i value(i) · eq(i) · scale` over the bound tables. Meaningful only
    /// when the schedule has an address phase.
    fn cycle_intermediate_claim(&self) -> F;

    /// The fully bound value coefficient — the reduction's final opening
    /// value (the advice/program-image polynomial's own opening; for the
    /// bytecode reduction, the chunk-weighted fold the per-chunk claims sum
    /// to). Errors while any variable remains unbound.
    fn final_claim(&self) -> Result<F, KernelError<F>>;

    /// The fully bound `aux` coefficients — the per-chunk `BytecodeChunk(i)`
    /// opening values. Empty for reductions without aux tables. Errors while
    /// any variable remains unbound.
    fn final_aux_claims(&self) -> Result<Vec<F>, KernelError<F>>;
}
