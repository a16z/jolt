//! Typed outputs produced by stage 6 verification.

use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage6Claims;
pub use super::outputs_a::{Stage6AddressPhasePublicOutput, VerifiedStage6AddressPhaseSumcheck};
pub use super::outputs_b::{
    AdviceCyclePhasePublicOutput, BooleanityPublicOutput, BytecodeReadRafPublicOutput,
    BytecodeReductionWeights, CommittedReductionCyclePhasePublicOutput,
    InstructionRaVirtualizationPublicOutput, RamRaVirtualizationPublicOutput,
    Stage6SumcheckPublicOutput, VerifiedAdviceCyclePhaseSumcheck, VerifiedBooleanitySumcheck,
    VerifiedBytecodeCyclePhaseSumcheck, VerifiedBytecodeReadRafSumcheck,
    VerifiedInstructionRaVirtualizationSumcheck, VerifiedProgramImageCyclePhaseSumcheck,
    VerifiedRamRaVirtualizationSumcheck, VerifiedStage6Sumcheck,
};

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
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage6PublicOutput<F>,
    /// Committed program mode only: bytecode claim-reduction batching
    /// challenge (core's `eta`).
    pub bytecode_reduction_eta: Option<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineStage6PublicOutput<F: Field> {
    pub field_inc_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ClearOutput<F: Field> {
    pub public: Stage6PublicOutput<F>,
    pub output_claims: Stage6Claims<F>,
    pub batch: VerifiedStage6Batch<F>,
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
    pub unsigned_inc_claim_reduction: Option<Stage6SumcheckPublicOutput<F>>,
    pub unsigned_inc_msb_booleanity: Option<Stage6SumcheckPublicOutput<F>>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: Stage6SumcheckPublicOutput<F>,
    pub trusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub untrusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub bytecode_cycle_phase: Option<CommittedReductionCyclePhasePublicOutput<F>>,
    pub program_image_cycle_phase: Option<CommittedReductionCyclePhasePublicOutput<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub inc_claim_reduction: Option<VerifiedStage6Sumcheck<F>>,
    pub unsigned_inc_claim_reduction: Option<VerifiedStage6Sumcheck<F>>,
    pub unsigned_inc_msb_booleanity: Option<VerifiedStage6Sumcheck<F>>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: VerifiedStage6Sumcheck<F>,
    pub trusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
    pub untrusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
    pub bytecode_cycle_phase: Option<VerifiedBytecodeCyclePhaseSumcheck<F>>,
    pub program_image_cycle_phase: Option<VerifiedProgramImageCyclePhaseSumcheck<F>>,
}
