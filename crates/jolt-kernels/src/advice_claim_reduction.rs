//! The advice claim-reduction slot: the two-phase reduction of a
//! trusted/untrusted advice opening (stage 6b cycle phase → stage 7 address
//! phase), plus the stage-4 advice opening evaluation it reduces.

use jolt_claims::protocols::jolt::{AdviceClaimReductionLayout, JoltAdviceKind};
use jolt_field::Field;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::precommitted_reduction::PrecommittedReductionProver;
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
