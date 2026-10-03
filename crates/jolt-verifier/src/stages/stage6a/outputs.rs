use jolt_claims::protocols::jolt::relations::booleanity::BooleanityAddressPhaseChallenges;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

pub use super::booleanity::BooleanityAddressPhaseOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafAddressPhaseOutputClaims;

use super::booleanity::BooleanityAddressPhase;
use super::bytecode_read_raf::BytecodeReadRafAddressPhase;

#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct Stage6aSumchecks<F: JoltField> {
    pub bytecode_read_raf: BytecodeReadRafAddressPhase<F>,
    pub booleanity: BooleanityAddressPhase<F>,
}

impl<F: JoltField> Stage6aOutputClaims<F> {
    pub fn from_base(
        bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims<F>,
        booleanity: BooleanityAddressPhaseOutputClaims<F>,
    ) -> Self {
        Self {
            #[cfg(feature = "field-inline")]
            bytecode_read_raf: bytecode_read_raf.into(),
            #[cfg(not(feature = "field-inline"))]
            bytecode_read_raf,
            booleanity,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage6aCarriedChallenges<F: JoltField> {
    /// The bytecode read-RAF address-phase draws (the fold gamma plus the five
    /// per-stage gammas), verbatim. Consumers folding with power vectors expand
    /// them via `stage_gamma_powers`.
    pub bytecode_read_raf: BytecodeReadRafAddressPhaseChallenges<F>,
    /// The booleanity address-phase draws (the reference address vector and
    /// the gamma), verbatim. The reference cycle is not carried: it is
    /// construction geometry (the reversed stage-5 instruction cycle, no draw
    /// of its own), rederived by its consumers from the stage-5 point.
    pub booleanity: BooleanityAddressPhaseChallenges<F>,
}

impl<F: JoltField> From<&Stage6aChallenges<F>> for Stage6aCarriedChallenges<F> {
    fn from(challenges: &Stage6aChallenges<F>) -> Self {
        Self {
            bytecode_read_raf: challenges.bytecode_read_raf,
            booleanity: challenges.booleanity.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage6aClearOutput<F: JoltField> {
    pub output_values: Stage6aOutputClaims<F>,
    pub output_points: Stage6aOutputPoints<F>,
    pub challenges: Stage6aCarriedChallenges<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6aZkOutput<F: JoltField, C> {
    pub challenges: Stage6aCarriedChallenges<F>,
    pub consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub output_claims: CommittedOutputClaimOutput<C>,
    pub output_points: Stage6aOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6aOutput<F: JoltField, C> {
    Clear(Stage6aClearOutput<F>),
    Zk(Stage6aZkOutput<F, C>),
}

impl<F: JoltField, C> Stage6aOutput<F, C> {
    pub fn output_points(&self) -> &Stage6aOutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    pub fn challenges(&self) -> &Stage6aCarriedChallenges<F> {
        match self {
            Self::Clear(output) => &output.challenges,
            Self::Zk(output) => &output.challenges,
        }
    }

    pub fn clear(&self) -> Result<&Stage6aClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage6a" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage6aZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => {
                Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6a" })
            }
        }
    }
}
