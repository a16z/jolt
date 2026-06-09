pub mod inputs;
pub mod manifest;
pub mod openings;
pub mod public_outputs;
pub mod stages;

use crate::support::{FixtureId, TestCase, VerifierPhase};

pub const CLEAR_INPUT_EVAL: TestCase = TestCase {
    name: "tamper_clear_opening_eval",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const CLEAR_INPUT_POINT: TestCase = TestCase {
    name: "tamper_clear_opening_point",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_INPUT_POINT: TestCase = TestCase {
    name: "tamper_zk_opening_point",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const CHECKED_INPUT_DIGEST: TestCase = TestCase {
    name: "tamper_checked_input_digest",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const VERIFIER_SETUP_DIGEST: TestCase = TestCase {
    name: "tamper_verifier_setup_digest",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const VERIFIER_SETUP_ARTIFACT: TestCase = TestCase {
    name: "tamper_verifier_setup_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_PROOF_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_proof_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_VMV_C_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_vmv_c_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_VMV_E1_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_vmv_e1_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_ZK_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_zk_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_DORY_E2_ARTIFACT: TestCase = TestCase {
    name: "tamper_zk_multiround_dory_e2_artifact",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_DORY_Y_COM_ARTIFACT: TestCase = TestCase {
    name: "tamper_zk_multiround_dory_y_com_artifact",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_DORY_SCALAR_PRODUCT_ARTIFACT: TestCase = TestCase {
    name: "tamper_zk_multiround_dory_scalar_product_artifact",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_REDUCE_ROUND_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_reduce_round_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_DORY_REDUCE_ROUND_ARTIFACT: TestCase = TestCase {
    name: "tamper_zk_multiround_dory_reduce_round_artifact",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_REDUCE_DIMENSIONS: TestCase = TestCase {
    name: "tamper_dory_reduce_dimensions",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_DORY_REDUCE_DIMENSIONS: TestCase = TestCase {
    name: "tamper_zk_multiround_dory_reduce_dimensions",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const GT_DIMENSIONS: TestCase = TestCase {
    name: "tamper_gt_dimensions",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const PACKING_DIMENSIONS: TestCase = TestCase {
    name: "tamper_packing_dimensions",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const DORY_FINAL_ARTIFACT: TestCase = TestCase {
    name: "tamper_dory_final_artifact",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const JOLT_COMMITMENT_CLAIM: TestCase = TestCase {
    name: "tamper_jolt_commitment_claim",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const JOLT_COMMITMENT_GT_CLAIM: TestCase = TestCase {
    name: "tamper_jolt_commitment_gt_claim",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const JOLT_EVALUATION_CLAIM: TestCase = TestCase {
    name: "tamper_jolt_evaluation_claim",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const TRANSCRIPT_SCALAR_CLAIM: TestCase = TestCase {
    name: "tamper_transcript_scalar_claim",
    zk: false,
    fixture: FixtureId::ClearInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const ZK_MULTIROUND_SIGMA_C_TRANSCRIPT_SCALAR_CLAIM: TestCase = TestCase {
    name: "tamper_zk_multiround_sigma_c_transcript_scalar_claim",
    zk: true,
    fixture: FixtureId::ZkInputMismatch,
    checked_at: VerifierPhase::CheckedInputs,
};

pub const STAGE1_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage1_payload",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_SUMCHECK_ROUNDS: TestCase = TestCase {
    name: "tamper_stage1_sumcheck_round_count",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_RELATION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_relation_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DIGIT_SELECTOR_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_digit_selector_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_SHIFT_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_shift_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_SHIFT_PUBLIC: TestCase = TestCase {
    name: "tamper_stage1_shift_public",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_BOUNDARY_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_boundary_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_BOUNDARY_PUBLIC: TestCase = TestCase {
    name: "tamper_stage1_boundary_public",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_MULTIPLICATION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_multiplication_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G1_SCALAR_MULTIPLICATION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g1_scalar_multiplication_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G1_SHIFT_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g1_shift_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G1_BOUNDARY_PUBLIC: TestCase = TestCase {
    name: "tamper_stage1_g1_boundary_public",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G1_ADDITION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g1_addition_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G2_SCALAR_MULTIPLICATION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g2_scalar_multiplication_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G2_SHIFT_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g2_shift_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G2_BOUNDARY_PUBLIC: TestCase = TestCase {
    name: "tamper_stage1_g2_boundary_public",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_G2_ADDITION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_g2_addition_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_LINE_STEP_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_line_step_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_LINE_EVALUATION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_line_evaluation_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_PAIR_PRODUCT_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_pair_product_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_ACCUMULATOR_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_accumulator_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_MILLER_BOUNDARY_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_miller_boundary_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_GT_TRANSITION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_gt_transition_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_G1_TRANSITION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_g1_transition_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_G2_TRANSITION_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_g2_transition_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_SCALAR_FOLD_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_scalar_fold_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_state_chain_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT: TestCase = TestCase {
    name: "tamper_stage1_dory_reduce_boundary_output",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const ZK_STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT: TestCase = TestCase {
    name: "tamper_zk_stage1_dory_reduce_state_chain_output",
    zk: true,
    fixture: FixtureId::ZkMultiround,
    checked_at: VerifierPhase::Stage1,
};

pub const ZK_STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT: TestCase = TestCase {
    name: "tamper_zk_stage1_dory_reduce_boundary_output",
    zk: true,
    fixture: FixtureId::ZkMultiround,
    checked_at: VerifierPhase::Stage1,
};

pub const STAGE2_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage2_payload",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_PUBLIC_VMV_C_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_public_vmv_c_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_PUBLIC_VMV_E1_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_public_vmv_e1_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_LINE_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_line_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_PAIR_PRODUCT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_pair_product_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_PAIR_PRODUCT_QUOTIENT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_pair_product_quotient_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_ACCUMULATOR_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_accumulator_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_ACCUMULATOR_QUOTIENT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_accumulator_quotient_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_BOUNDARY_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_boundary_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_G1_SHIFT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_g1_shift_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_G1_BOUNDARY_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_g1_boundary_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_G2_SHIFT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_g2_shift_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_G2_BOUNDARY_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_g2_boundary_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_SCALAR_FOLD_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_scalar_fold_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_INITIAL_STATE_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_initial_state_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_PROOF_ARTIFACT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_proof_artifact_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_SETUP_ARTIFACT_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_setup_artifact_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_TRANSCRIPT_SCALAR_COPY_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_transcript_scalar_copy_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE: TestCase = TestCase {
    name: "tamper_stage2_dory_reduce_public_fold_value",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage2,
};

pub const ZK_STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE: TestCase = TestCase {
    name: "tamper_zk_stage2_dory_reduce_public_fold_value",
    zk: true,
    fixture: FixtureId::ZkMultiround,
    checked_at: VerifierPhase::Stage2,
};

pub const STAGE3_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage3_payload",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage3,
};

pub const STAGE3_REDUCED_OPENINGS: TestCase = TestCase {
    name: "tamper_stage3_reduced_openings",
    zk: false,
    fixture: FixtureId::StagePayloadMismatch,
    checked_at: VerifierPhase::Stage3,
};

pub const OPENING_CLAIM_POINT: TestCase = TestCase {
    name: "tamper_opening_claim_point",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    checked_at: VerifierPhase::Opening,
};

pub const OPENING_CLAIM_EVAL: TestCase = TestCase {
    name: "tamper_opening_claim_eval",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    checked_at: VerifierPhase::Opening,
};

pub const HYRAX_OPENING_ROW: TestCase = TestCase {
    name: "tamper_hyrax_opening_row",
    zk: false,
    fixture: FixtureId::HyraxOpeningMismatch,
    checked_at: VerifierPhase::Opening,
};

pub const HYRAX_OPENING_SCALAR: TestCase = TestCase {
    name: "tamper_hyrax_opening_scalar",
    zk: false,
    fixture: FixtureId::HyraxOpeningMismatch,
    checked_at: VerifierPhase::Opening,
};

pub const DENSE_COMMITMENT: TestCase = TestCase {
    name: "tamper_dense_commitment",
    zk: false,
    fixture: FixtureId::DenseCommitmentMismatch,
    checked_at: VerifierPhase::Opening,
};

pub const PUBLIC_OUTPUT: TestCase = TestCase {
    name: "tamper_public_output",
    zk: false,
    fixture: FixtureId::PublicOutputMismatch,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ZK_PUBLIC_OUTPUT: TestCase = TestCase {
    name: "tamper_zk_public_output",
    zk: true,
    fixture: FixtureId::ZkPublicOutputMismatch,
    checked_at: VerifierPhase::NativeOutput,
};

pub const NATIVE_FINAL_INPUT: TestCase = TestCase {
    name: "tamper_native_final_input",
    zk: false,
    fixture: FixtureId::NativeFinalInputMismatch,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ZK_NATIVE_FINAL_INPUT: TestCase = TestCase {
    name: "tamper_zk_native_final_input",
    zk: true,
    fixture: FixtureId::ZkNativeFinalInputMismatch,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ALL: &[TestCase] = &[
    CLEAR_INPUT_EVAL,
    CLEAR_INPUT_POINT,
    ZK_INPUT_POINT,
    CHECKED_INPUT_DIGEST,
    VERIFIER_SETUP_DIGEST,
    VERIFIER_SETUP_ARTIFACT,
    DORY_PROOF_ARTIFACT,
    DORY_VMV_C_ARTIFACT,
    DORY_VMV_E1_ARTIFACT,
    DORY_ZK_ARTIFACT,
    ZK_MULTIROUND_DORY_E2_ARTIFACT,
    ZK_MULTIROUND_DORY_Y_COM_ARTIFACT,
    ZK_MULTIROUND_DORY_SCALAR_PRODUCT_ARTIFACT,
    DORY_REDUCE_ROUND_ARTIFACT,
    ZK_MULTIROUND_DORY_REDUCE_ROUND_ARTIFACT,
    DORY_REDUCE_DIMENSIONS,
    ZK_MULTIROUND_DORY_REDUCE_DIMENSIONS,
    GT_DIMENSIONS,
    PACKING_DIMENSIONS,
    DORY_FINAL_ARTIFACT,
    JOLT_COMMITMENT_CLAIM,
    JOLT_COMMITMENT_GT_CLAIM,
    JOLT_EVALUATION_CLAIM,
    TRANSCRIPT_SCALAR_CLAIM,
    ZK_MULTIROUND_SIGMA_C_TRANSCRIPT_SCALAR_CLAIM,
    STAGE1_PAYLOAD,
    STAGE1_SUMCHECK_ROUNDS,
    STAGE1_RELATION_OUTPUT,
    STAGE1_DIGIT_SELECTOR_OUTPUT,
    STAGE1_SHIFT_OUTPUT,
    STAGE1_SHIFT_PUBLIC,
    STAGE1_BOUNDARY_OUTPUT,
    STAGE1_BOUNDARY_PUBLIC,
    STAGE1_MULTIPLICATION_OUTPUT,
    STAGE1_G1_SCALAR_MULTIPLICATION_OUTPUT,
    STAGE1_G1_SHIFT_OUTPUT,
    STAGE1_G1_BOUNDARY_PUBLIC,
    STAGE1_G1_ADDITION_OUTPUT,
    STAGE1_G2_SCALAR_MULTIPLICATION_OUTPUT,
    STAGE1_G2_SHIFT_OUTPUT,
    STAGE1_G2_BOUNDARY_PUBLIC,
    STAGE1_G2_ADDITION_OUTPUT,
    STAGE1_LINE_STEP_OUTPUT,
    STAGE1_LINE_EVALUATION_OUTPUT,
    STAGE1_PAIR_PRODUCT_OUTPUT,
    STAGE1_ACCUMULATOR_OUTPUT,
    STAGE1_MILLER_BOUNDARY_OUTPUT,
    STAGE1_DORY_REDUCE_GT_TRANSITION_OUTPUT,
    STAGE1_DORY_REDUCE_G1_TRANSITION_OUTPUT,
    STAGE1_DORY_REDUCE_G2_TRANSITION_OUTPUT,
    STAGE1_DORY_REDUCE_SCALAR_FOLD_OUTPUT,
    STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT,
    STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT,
    ZK_STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT,
    ZK_STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT,
    STAGE2_PAYLOAD,
    STAGE2_COPY_VALUE,
    STAGE2_PUBLIC_VMV_C_COPY_VALUE,
    STAGE2_PUBLIC_VMV_E1_COPY_VALUE,
    STAGE2_LINE_COPY_VALUE,
    STAGE2_PAIR_PRODUCT_COPY_VALUE,
    STAGE2_PAIR_PRODUCT_QUOTIENT_COPY_VALUE,
    STAGE2_ACCUMULATOR_COPY_VALUE,
    STAGE2_ACCUMULATOR_QUOTIENT_COPY_VALUE,
    STAGE2_BOUNDARY_COPY_VALUE,
    STAGE2_G1_SHIFT_COPY_VALUE,
    STAGE2_G1_BOUNDARY_COPY_VALUE,
    STAGE2_G2_SHIFT_COPY_VALUE,
    STAGE2_G2_BOUNDARY_COPY_VALUE,
    STAGE2_DORY_REDUCE_SCALAR_FOLD_COPY_VALUE,
    STAGE2_DORY_REDUCE_INITIAL_STATE_COPY_VALUE,
    STAGE2_DORY_REDUCE_PROOF_ARTIFACT_COPY_VALUE,
    STAGE2_DORY_REDUCE_SETUP_ARTIFACT_COPY_VALUE,
    STAGE2_DORY_REDUCE_TRANSCRIPT_SCALAR_COPY_VALUE,
    STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE,
    ZK_STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE,
    STAGE3_PAYLOAD,
    STAGE3_REDUCED_OPENINGS,
    OPENING_CLAIM_POINT,
    OPENING_CLAIM_EVAL,
    HYRAX_OPENING_ROW,
    HYRAX_OPENING_SCALAR,
    DENSE_COMMITMENT,
    PUBLIC_OUTPUT,
    ZK_PUBLIC_OUTPUT,
    NATIVE_FINAL_INPUT,
    ZK_NATIVE_FINAL_INPUT,
];
