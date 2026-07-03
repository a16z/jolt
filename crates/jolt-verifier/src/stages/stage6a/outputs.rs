//! Typed inputs consumed and outputs produced by stage 6a (address-phase)
//! verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

// The per-relation produced-claim structs live in their relation modules
// (cell-generic, `#[derive(OutputClaims)]`); re-export them so consumers and the
// generated stage-6a aggregates keep resolving them through `stage6a::outputs`.
pub use super::booleanity::BooleanityAddressPhaseOutputClaims;
pub use super::bytecode_read_raf::BytecodeReadRafAddressPhaseOutputClaims;

use super::booleanity::BooleanityAddressPhase;
use super::bytecode_read_raf::BytecodeReadRafAddressPhase;

/// Source-of-truth for stage 6a's two-instance address-phase sumcheck batch
/// (bytecode read-RAF, booleanity). `#[derive(SumcheckBatch)]` generates the
/// `Stage6a{Input,Output}{Claims,Points}<F>` and `Stage6aChallenges<F>`
/// aggregates — one field per instance, in this declaration order — plus the
/// batched-verify drivers. No alias dedup in the address phase, so the generated
/// absorb (`append_output_claims`; member order: bytecode read-RAF's
/// `intermediate` then `val_stages`, then booleanity's `intermediate`) is the
/// canonical Fiat-Shamir order.
///
/// The bytecode read-RAF member's wire set extends its output `Expr` with the
/// committed-program-only staged `BytecodeValStage` openings (see its
/// `wire_output_openings` override), so the generated `output_shape`
/// count/validator cover the val-stage presence and count.
#[derive(SumcheckBatch)]
#[sumcheck_batch(output_shape)]
pub struct Stage6aSumchecks<F: Field> {
    pub bytecode_read_raf: BytecodeReadRafAddressPhase<F>,
    pub booleanity: BooleanityAddressPhase<F>,
}

/// The stage-6a Fiat-Shamir draws sampled before/around the address-phase batch
/// but consumed only by stage 6b. The prover's booleanity subprotocol samples its
/// gamma (and the reference-address padding) before the 6a batch runs, and the
/// per-stage folding gammas are drawn with the 6a batch's bytecode fold; only
/// stage 6b's members consume them, so 6a carries them downstream as typed
/// upstream values (the same idiom as `Stage2ZkOutput`'s `product_tau_high`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6aCarriedChallenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6aClearOutput<F: Field> {
    /// The produced address-phase opening *points*, read by stage 6b to construct
    /// the cycle-phase batch.
    pub output_points: Stage6aOutputPoints<F>,
    /// The pre-/around-batch draws carried to stage 6b (see
    /// [`Stage6aCarriedChallenges`]).
    pub challenges: Stage6aCarriedChallenges<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6aZkOutput<F: Field, C> {
    pub challenges: Stage6aCarriedChallenges<F>,
    pub consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening *points*, the ZK counterpart of the clear path's
    /// `output_points`. Read by stage 6b and BlindFold.
    pub output_points: Stage6aOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6aOutput<F: Field, C> {
    Clear(Stage6aClearOutput<F>),
    Zk(Stage6aZkOutput<F, C>),
}

impl<F: Field, C> Stage6aOutput<F, C> {
    /// The produced address-phase opening *points*, available regardless of mode.
    pub fn output_points(&self) -> &Stage6aOutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    /// The pre-/around-batch draws carried to stage 6b, available in both modes.
    pub fn challenges(&self) -> &Stage6aCarriedChallenges<F> {
        match self {
            Self::Clear(output) => &output.challenges,
            Self::Zk(output) => &output.challenges,
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
