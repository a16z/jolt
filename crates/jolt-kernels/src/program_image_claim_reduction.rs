//! The program-image claim-reduction slot (stage 6b cycle phase → stage 7
//! address phase): reduces the stage-4 `ProgramImageInitContributionRw`
//! contribution into a final `ProgramImageInit` opening over the shared
//! precommitted schedule.

use jolt_claims::protocols::jolt::ProgramImageClaimReductionLayout;
use jolt_field::Field;

use crate::precommitted_reduction::PrecommittedReductionProver;
use crate::{KernelError, ProofSession};

/// The program-image claim-reduction slot. `r_addr_rw` is the stage-2 RAM
/// read-write address point (the staged contribution's point);
/// `bytecode_words` the prover-retained RAM-remapped image words;
/// `start_index` the image block's RAM word offset.
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
