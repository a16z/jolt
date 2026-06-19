//! Stage 6 verifier entry point.

pub mod booleanity;
pub mod bytecode_read_raf;
pub mod inc_claim_reduction;
pub mod inputs;
pub mod instruction_ra_virtualization;
pub mod outputs;
pub mod ram_hamming_booleanity;
pub mod ram_ra_virtualization;
pub mod verify;

pub use booleanity::{
    Booleanity, BooleanityAddressPhase, BooleanityAddressPhaseInputClaims,
    BooleanityAddressPhaseOutputClaims, BooleanityInputClaims, BooleanityOutputClaims,
};
pub use bytecode_read_raf::{
    BytecodeReadRaf, BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims, BytecodeReadRafCycleInputs,
    BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims,
};
pub use inc_claim_reduction::{
    IncClaimReduction, IncClaimReductionInputClaims, IncClaimReductionOutputClaims,
};
pub use instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationInputClaims,
    InstructionRaVirtualizationOutputClaims,
};
pub use ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityInputClaims, RamHammingBooleanityOutputClaims,
};
pub use ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationInputClaims, RamRaVirtualizationOutputClaims,
};
pub use inputs::{deps, Deps};
pub use outputs::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput};
pub use verify::{
    append_stage6_opening_claims, stage6_advice_cycle_phase_expected_output,
    stage6_advice_cycle_phase_reference, stage6_advice_cycle_phase_verified,
    stage6_batch_input_claims, stage6_batch_points, stage6_booleanity_expected_output,
    stage6_booleanity_reference, stage6_bytecode_cycle_points, stage6_bytecode_gamma_count,
    stage6_bytecode_ra_point, stage6_bytecode_read_raf_address_input,
    stage6_bytecode_read_raf_expected_output, stage6_bytecode_read_raf_output_coefficient,
    stage6_bytecode_read_raf_point, stage6_bytecode_register_points, stage6_clear_output,
    stage6_expected_final_claim, stage6_expected_output_claim_values,
    stage6_inc_claim_reduction_cycle_points, stage6_inc_claim_reduction_expected_output,
    stage6_input_claim_values, stage6_instruction_ra_virtualization_expected_output,
    stage6_instruction_read_raf_point, stage6_output_claim_values,
    stage6_post_address_transcript_challenges, stage6_pre_address_transcript_challenges,
    stage6_public_output, stage6_ram_hamming_booleanity_expected_output,
    stage6_ram_ra_virtualization_expected_output, stage6_ram_reduced_opening_point,
    stage6_stage1_cycle_binding, stage6_stage1_gamma_count, stage6_stage2_gamma_count,
    stage6_stage3_gamma_count, stage6_stage4_gamma_count, stage6_stage5_gamma_count,
    stage6_stage5_ram_reduced_opening_point, stage6_zk_instruction_read_raf_point,
    stage6_zk_stage5_ram_reduced_opening_point, verify, Stage6AdviceCyclePhaseReference,
    Stage6BatchExpectedOutputClaims, Stage6BatchInputClaims, Stage6BatchPointContext,
    Stage6BatchPointInputs, Stage6BatchPoints, Stage6BooleanityExpectedOutputInputs,
    Stage6BooleanityReference, Stage6BytecodeRaPoint, Stage6BytecodeReadRafExpectedOutputInputs,
    Stage6BytecodeReadRafOutputCoefficientInputs, Stage6BytecodeReadRafPoint,
    Stage6BytecodeRegisterPoints, Stage6ClearOutputRequest, Stage6IncClaimReductionCyclePoints,
    Stage6IncClaimReductionExpectedOutputInputs, Stage6InputClaimChallengeValues,
    Stage6InstructionRaVirtualizationExpectedOutputInputs, Stage6InstructionReadRafPoint,
    Stage6PostAddressChallenges, Stage6PreAddressChallenges,
    Stage6RamRaVirtualizationExpectedOutputInputs, Stage6RamReducedOpeningPoint,
    Stage6TranscriptChallenges,
};
