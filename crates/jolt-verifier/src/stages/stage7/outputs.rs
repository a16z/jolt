//! Typed outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial};
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage7Claims;

/// Final opening of a precommitted polynomial, resolved from whichever stage
/// completed its claim reduction (stage 6b cycle phase or stage 7 address
/// phase). Stage 8 consumes these as anchors and batch members of the final
/// PCS opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedFinalOpening<F: Field> {
    pub polynomial: JoltCommittedPolynomial,
    pub point: Vec<F>,
    /// `None` in ZK mode, where opening claims stay committed.
    pub opening_claim: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub hamming_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ClearOutput<F: Field> {
    pub public: Stage7PublicOutput<F>,
    pub output_claims: Stage7Claims<F>,
    pub batch: VerifiedStage7Batch<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: Field, C> {
    pub public: Stage7PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionPublicOutput<F>,
    pub trusted_advice_address_phase: Option<AdviceAddressPhasePublicOutput<F>>,
    pub untrusted_advice_address_phase: Option<AdviceAddressPhasePublicOutput<F>>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7Output<F: Field, C> {
    Clear(Stage7ClearOutput<F>),
    Zk(Stage7ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage7Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub hamming_weight_claim_reduction: VerifiedHammingWeightClaimReductionSumcheck<F>,
    pub trusted_advice_address_phase: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
    pub untrusted_advice_address_phase: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedHammingWeightClaimReductionSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HammingWeightClaimReductionPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedAdviceAddressPhaseSumcheck<F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceAddressPhasePublicOutput<F: Field> {
    pub kind: JoltAdviceKind,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
}
