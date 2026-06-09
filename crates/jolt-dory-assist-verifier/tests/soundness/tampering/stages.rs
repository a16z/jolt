use crate::support::{
    assert_accepts, assert_rejects, assert_rejects_at_stage, clear_base_case,
    clear_multiround_case, clear_shift_public_kernel_case, tamper_stage1_accumulator_output,
    tamper_stage1_boundary_output, tamper_stage1_boundary_public,
    tamper_stage1_digit_selector_output, tamper_stage1_dory_reduce_boundary_output,
    tamper_stage1_dory_reduce_g1_transition_output, tamper_stage1_dory_reduce_g2_transition_output,
    tamper_stage1_dory_reduce_gt_transition_output, tamper_stage1_dory_reduce_scalar_fold_output,
    tamper_stage1_dory_reduce_state_chain_output, tamper_stage1_g1_addition_output,
    tamper_stage1_g1_boundary_public, tamper_stage1_g1_scalar_multiplication_output,
    tamper_stage1_g1_shift_output, tamper_stage1_g2_addition_output,
    tamper_stage1_g2_boundary_public, tamper_stage1_g2_scalar_multiplication_output,
    tamper_stage1_g2_shift_output, tamper_stage1_line_evaluation_output,
    tamper_stage1_line_step_output, tamper_stage1_miller_boundary_output,
    tamper_stage1_multiplication_output, tamper_stage1_pair_product_output, tamper_stage1_payload,
    tamper_stage1_relation_output, tamper_stage1_shift_output, tamper_stage1_shift_public,
    tamper_stage1_sumcheck_round_count, tamper_stage2_accumulator_copy_value,
    tamper_stage2_accumulator_quotient_copy_value, tamper_stage2_boundary_copy_value,
    tamper_stage2_copy_value, tamper_stage2_dory_reduce_initial_state_copy_value,
    tamper_stage2_dory_reduce_proof_artifact_copy_value,
    tamper_stage2_dory_reduce_public_fold_value, tamper_stage2_dory_reduce_scalar_fold_copy_value,
    tamper_stage2_dory_reduce_setup_artifact_copy_value,
    tamper_stage2_dory_reduce_transcript_scalar_copy_value, tamper_stage2_g1_boundary_copy_value,
    tamper_stage2_g1_shift_copy_value, tamper_stage2_g2_boundary_copy_value,
    tamper_stage2_g2_shift_copy_value, tamper_stage2_line_copy_value,
    tamper_stage2_pair_product_copy_value, tamper_stage2_pair_product_quotient_copy_value,
    tamper_stage2_payload, tamper_stage2_public_vmv_c_copy_value,
    tamper_stage2_public_vmv_e1_copy_value, tamper_stage2_zk_dory_reduce_public_fold_value,
    tamper_stage3_payload, tamper_stage3_reduced_openings, zk_multiround_case,
};
use jolt_dory_assist_verifier::DoryAssistStage;

#[test]
fn tampered_stage1_payload_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_payload(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_sumcheck_round_count_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_sumcheck_round_count(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_relation_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_relation_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_digit_selector_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_digit_selector_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_shift_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_shift_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn stage1_shift_public_kernel_fixture_accepts() {
    assert_accepts(clear_shift_public_kernel_case().verify_clear());
}

#[test]
fn tampered_stage1_shift_public_rejects() {
    let mut case = clear_shift_public_kernel_case();
    tamper_stage1_shift_public(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_boundary_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_boundary_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_boundary_public_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_boundary_public(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_multiplication_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_multiplication_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g1_scalar_multiplication_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g1_scalar_multiplication_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g1_shift_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g1_shift_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g1_boundary_public_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g1_boundary_public(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g1_addition_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g1_addition_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g2_scalar_multiplication_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g2_scalar_multiplication_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g2_shift_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g2_shift_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g2_boundary_public_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g2_boundary_public(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_g2_addition_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_g2_addition_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_line_step_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_line_step_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_line_evaluation_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_line_evaluation_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_pair_product_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_pair_product_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_accumulator_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_accumulator_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_miller_boundary_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_miller_boundary_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_dory_reduce_gt_transition_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_dory_reduce_gt_transition_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_dory_reduce_g1_transition_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_dory_reduce_g1_transition_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_dory_reduce_g2_transition_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_dory_reduce_g2_transition_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_dory_reduce_scalar_fold_output_rejects() {
    let mut case = clear_base_case();
    tamper_stage1_dory_reduce_scalar_fold_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage1_dory_reduce_state_chain_output_rejects_for_multiround() {
    let mut case = clear_multiround_case();
    tamper_stage1_dory_reduce_state_chain_output(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage1);
}

#[test]
fn tampered_stage1_dory_reduce_boundary_output_rejects_for_multiround() {
    let mut case = clear_multiround_case();
    tamper_stage1_dory_reduce_boundary_output(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage1);
}

#[test]
fn tampered_zk_stage1_dory_reduce_state_chain_output_rejects_for_multiround() {
    let mut case = zk_multiround_case();
    tamper_stage1_dory_reduce_state_chain_output(&mut case);
    assert_rejects_at_stage(case.verify_zk(), DoryAssistStage::Stage1);
}

#[test]
fn tampered_zk_stage1_dory_reduce_boundary_output_rejects_for_multiround() {
    let mut case = zk_multiround_case();
    tamper_stage1_dory_reduce_boundary_output(&mut case);
    assert_rejects_at_stage(case.verify_zk(), DoryAssistStage::Stage1);
}

#[test]
fn tampered_stage2_payload_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_payload(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_public_vmv_c_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_public_vmv_c_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_public_vmv_e1_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_public_vmv_e1_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_line_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_line_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_pair_product_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_pair_product_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_pair_product_quotient_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_pair_product_quotient_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_accumulator_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_accumulator_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_accumulator_quotient_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_accumulator_quotient_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_boundary_copy_value_rejects() {
    let mut case = clear_base_case();
    tamper_stage2_boundary_copy_value(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage2_g1_shift_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_g1_shift_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_g1_boundary_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_g1_boundary_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_g2_shift_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_g2_shift_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_g2_boundary_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_g2_boundary_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_scalar_fold_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_dory_reduce_scalar_fold_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_initial_state_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_dory_reduce_initial_state_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_proof_artifact_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_dory_reduce_proof_artifact_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_setup_artifact_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_dory_reduce_setup_artifact_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_transcript_scalar_copy_value_rejects_at_stage2() {
    let mut case = clear_base_case();
    tamper_stage2_dory_reduce_transcript_scalar_copy_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage2_dory_reduce_public_fold_value_rejects_at_stage2_for_multiround() {
    let mut case = clear_multiround_case();
    tamper_stage2_dory_reduce_public_fold_value(&mut case);
    assert_rejects_at_stage(case.verify_clear(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_zk_stage2_dory_reduce_public_fold_value_rejects_at_stage2_for_multiround() {
    let mut case = zk_multiround_case();
    tamper_stage2_zk_dory_reduce_public_fold_value(&mut case);
    assert_rejects_at_stage(case.verify_zk(), DoryAssistStage::Stage2);
}

#[test]
fn tampered_stage3_payload_rejects() {
    let mut case = clear_base_case();
    tamper_stage3_payload(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_stage3_reduced_openings_rejects() {
    let mut case = clear_base_case();
    tamper_stage3_reduced_openings(&mut case);
    assert_rejects(case.verify_clear());
}
