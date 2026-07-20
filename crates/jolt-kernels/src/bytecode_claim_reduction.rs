//! The committed-bytecode claim-reduction slot (stage 6b cycle phase →
//! stage 7 address phase): reduces the five staged `BytecodeValStage(i)`
//! claims into per-chunk `BytecodeChunk(i)` openings.

use jolt_claims::protocols::jolt::BytecodeClaimReductionLayout;
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;

use crate::precommitted_reduction::PrecommittedReductionProver;
use crate::{KernelError, ProofSession};

/// The committed-bytecode claim-reduction slot. `weights` are the public
/// chunk/lane weights the recipe built with the verifier's own promoted
/// `bytecode_reduction_weights`; `bytecode` is the prover-retained full
/// bytecode the chunk grids materialize from.
pub trait BytecodeClaimReduction<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        layout: &BytecodeClaimReductionLayout,
        weights: &BytecodeReductionWeights<F>,
        bytecode: &[JoltInstructionRow],
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}
