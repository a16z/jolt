//! Typed outputs produced by stage 6 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage6OutputClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PublicOutput<F: Field> {
    pub address_phase_challenges: Vec<F>,
    pub address_phase_batching_coefficients: Vec<F>,
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub inc_gamma: F,
    /// Committed program mode only: bytecode claim-reduction batching
    /// challenge (core's `eta`).
    pub bytecode_reduction_eta: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ClearOutput<F: Field> {
    pub public: Stage6PublicOutput<F>,
    /// The produced opening *values* (wire form); read by later stages and the
    /// Fiat-Shamir opening-claim encoder.
    pub output_claims: Stage6OutputClaims<F>,
    /// The produced opening *points* (point-only cell), paired field-for-field with
    /// `output_claims`. Stages 7 and 8 read each relation's opening point off these
    /// cells (via the `Stage6OutputClaims<Vec<F>>` accessors), replacing the
    /// per-relation points the retired `VerifiedStage6Batch` carried.
    pub output_points: Stage6OutputClaims<Vec<F>>,
    pub batch: VerifiedStage6Batch<F>,
}

impl<F: Field> Stage6ClearOutput<F> {
    pub const fn advice_cycle_phase(
        &self,
        kind: JoltAdviceKind,
    ) -> Option<&VerifiedAdviceCyclePhaseSumcheck<F>> {
        match kind {
            JoltAdviceKind::Trusted => self.batch.trusted_advice_cycle_phase.as_ref(),
            JoltAdviceKind::Untrusted => self.batch.untrusted_advice_cycle_phase.as_ref(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ZkOutput<F: Field, C> {
    pub public: Stage6PublicOutput<F>,
    pub address_phase_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub address_phase_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub bytecode_read_raf_address: Stage6AddressPhasePublicOutput<F>,
    pub booleanity_address: Stage6AddressPhasePublicOutput<F>,
    pub bytecode_read_raf: BytecodeReadRafPublicOutput<F>,
    pub booleanity: BooleanityPublicOutput<F>,
    pub ram_hamming_booleanity: Stage6SumcheckPublicOutput<F>,
    pub ram_ra_virtualization: RamRaVirtualizationPublicOutput<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationPublicOutput<F>,
    pub inc_claim_reduction: Stage6SumcheckPublicOutput<F>,
    pub trusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub untrusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub bytecode_cycle_phase: Option<CommittedReductionCyclePhasePublicOutput<F>>,
    pub program_image_cycle_phase: Option<CommittedReductionCyclePhasePublicOutput<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
// Transitional: `Stage6ClearOutput` carries both the retiring `batch` and the new
// `output_points` while consumers migrate; the follow-up commit removes `batch`,
// shrinking the `Clear` variant back under the threshold.
#[expect(clippy::large_enum_variant)]
pub enum Stage6Output<F: Field, C> {
    Clear(Stage6ClearOutput<F>),
    Zk(Stage6ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6Batch<F: Field> {
    pub address_phase_batching_coefficients: Vec<F>,
    pub address_phase_sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub address_phase_sumcheck_final_claim: F,
    pub address_phase_expected_final_claim: F,
    pub bytecode_read_raf_address: VerifiedStage6AddressPhaseSumcheck<F>,
    pub booleanity_address: VerifiedStage6AddressPhaseSumcheck<F>,
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub bytecode_read_raf: VerifiedBytecodeReadRafSumcheck<F>,
    pub booleanity: VerifiedBooleanitySumcheck<F>,
    pub ram_hamming_booleanity: VerifiedStage6Sumcheck<F>,
    pub ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck<F>,
    pub instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck<F>,
    pub inc_claim_reduction: VerifiedStage6Sumcheck<F>,
    pub trusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
    pub untrusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
    pub bytecode_cycle_phase: Option<VerifiedBytecodeCyclePhaseSumcheck<F>>,
    pub program_image_cycle_phase: Option<VerifiedProgramImageCyclePhaseSumcheck<F>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6AddressPhaseSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6AddressPhasePublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBytecodeReadRafSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBooleanitySumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamRaVirtualizationSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedInstructionRaVirtualizationSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedAdviceCyclePhaseSumcheck<F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceCyclePhasePublicOutput<F: Field> {
    pub kind: JoltAdviceKind,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
}

/// Public bytecode claim-reduction state shared by the cycle and address
/// phases: the per-chunk weights over dropped address bits, the chunk-local
/// cycle point, and the gamma-folded lane weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReductionWeights<F: Field> {
    pub r_bc: Vec<F>,
    pub chunk_rbc_weights: Vec<F>,
    pub lane_weights: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBytecodeCyclePhaseSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
    pub weights: BytecodeReductionWeights<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedProgramImageCyclePhaseSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
    pub expected_output_claim: F,
}

/// Cycle phase of the committed bytecode or program-image claim reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedReductionCyclePhasePublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
}
