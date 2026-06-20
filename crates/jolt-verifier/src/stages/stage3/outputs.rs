//! Typed outputs produced by stage 3 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::OpeningClaim;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage3OutputClaims;

/// The Fiat-Shamir challenges the verifier draws during stage 3: the three
/// per-relation batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Challenges<F: Field> {
    pub shift_gamma: F,
    pub instruction_gamma: F,
    pub registers_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ClearOutput<F: Field> {
    pub challenges: Stage3Challenges<F>,
    /// The produced stage-3 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell. The opening points are derived from each relation's
    /// sumcheck point; pairing them with the values here lets later stages consume a
    /// ready `OpeningClaim` instead of re-joining a value with a separately-tracked
    /// point.
    pub output_claims: Stage3OutputClaims<OpeningClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ZkOutput<F: Field, C> {
    pub challenges: Stage3Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
}

// The clear variant carries the located opening claims (point + value) that
// stages 4 and 6 read on the hot path; the ZK variant carries only committed
// consistency. Boxing the common clear variant to shrink the rarer ZK one would
// add indirection to every clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3Output<F: Field, C> {
    Clear(Stage3ClearOutput<F>),
    Zk(Stage3ZkOutput<F, C>),
}

impl<F: Field, C> Stage3Output<F, C> {
    pub fn clear(&self) -> Result<&Stage3ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage3" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage3ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage3" }),
        }
    }
}
