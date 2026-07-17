//! The precommitted claim-reduction family: the two-phase (stage-6b cycle
//! phase → stage-7 address phase) instance trait and the advice,
//! committed-bytecode, and program-image slot traits that return it.

use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind,
    ProgramImageClaimReductionLayout,
};
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession};

/// The advice claim-reduction slot: the stage-4 opening evaluation and the
/// stage-6b/7 reduction member share it because both are the advice
/// polynomial's protocol duties (there is exactly one advice oracle read
/// path).
pub trait AdviceClaimReduction<F: Field> {
    /// Evaluate the advice polynomial at `point` (big-endian) — the value the
    /// stage-4 RAM value-check stages under `@RamValCheck` for this kind.
    fn evaluate(
        &self,
        session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<F, KernelError<F>>;

    /// Build the two-phase reduction member for `kind`. `r_val` is the staged
    /// stage-4 opening point (big-endian, `advice_vars` long) the eq table is
    /// built from.
    fn prepare(
        &self,
        session: &mut ProofSession,
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        r_val: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}

/// The committed-bytecode claim-reduction slot (reduces the five staged
/// `BytecodeValStage(i)` claims into per-chunk `BytecodeChunk(i)` openings).
/// `weights` are the public chunk/lane weights the recipe built with the
/// verifier's own promoted `bytecode_reduction_weights`; `bytecode` is the
/// prover-retained full bytecode the chunk grids materialize from.
pub trait BytecodeClaimReduction<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        layout: &BytecodeClaimReductionLayout,
        weights: &BytecodeReductionWeights<F>,
        bytecode: &[JoltInstructionRow],
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}

/// The program-image claim-reduction slot (reduces the stage-4
/// `ProgramImageInitContributionRw` contribution into a final
/// `ProgramImageInit` opening over the shared precommitted schedule).
/// `r_addr_rw` is the stage-2 RAM read-write address point (the staged
/// contribution's point); `bytecode_words` the prover-retained RAM-remapped
/// image words; `start_index` the image block's RAM word offset.
pub trait ProgramImageClaimReduction<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        layout: &ProgramImageClaimReductionLayout,
        r_addr_rw: &[F],
        start_index: usize,
        bytecode_words: &[u64],
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}

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
