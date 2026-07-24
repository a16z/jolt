//! The registers value-evaluation (stage 5) slot.

use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-5 registers value-evaluation slot.
pub trait RegistersValEvaluationProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        registers_val_point: &[F],
        challenges: &NoChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>;
}
