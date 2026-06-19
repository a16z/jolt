//! Typed outputs produced by stage 6 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;
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
    /// cells (via the `Stage6OutputClaims<Vec<F>>` accessors).
    pub output_points: Stage6OutputClaims<Vec<F>>,
    /// Committed-program mode only: the bytecode claim-reduction's per-chunk
    /// weights (`r_bc`, chunk weights, gamma-folded lane weights). These are
    /// public derived data (not openings), so stage 7's bytecode address phase
    /// reads them here rather than recomputing them.
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
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

// The ZK variant is the larger one (it carries the committed sumcheck
// consistency data); the enum is a transient stage output, matched and consumed
// immediately rather than stored in bulk, so boxing the rare ZK variant would add
// indirection without a meaningful size win.
#[expect(
    clippy::large_enum_variant,
    reason = "transient stage output; the larger ZK variant is matched immediately, not stored in bulk"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6Output<F: Field, C> {
    Clear(Stage6ClearOutput<F>),
    Zk(Stage6ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6AddressPhasePublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
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
pub struct BooleanityPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
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

/// Cycle phase of the committed bytecode or program-image claim reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedReductionCyclePhasePublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
}
