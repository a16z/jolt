use crate::support::{
    assert_rejects, clear_base_case, tamper_checked_input_digest, tamper_clear_eval,
    tamper_dory_final_artifact, tamper_dory_proof_artifact, tamper_dory_reduce_dimensions,
    tamper_dory_reduce_round_artifact, tamper_dory_scalar_product_artifact,
    tamper_dory_vmv_c_artifact, tamper_dory_vmv_e1_artifact, tamper_dory_zk_artifact,
    tamper_dory_zk_y_com_artifact, tamper_gt_dimensions, tamper_jolt_commitment_claim,
    tamper_jolt_commitment_gt_claim, tamper_jolt_evaluation_claim, tamper_opening_point,
    tamper_packing_dimensions, tamper_transcript_scalar_claim, tamper_verifier_setup_artifact,
    tamper_verifier_setup_digest, tamper_zk_sigma_c_transcript_scalar_claim, zk_base_case,
    zk_multiround_case,
};

#[test]
fn tampered_clear_opening_eval_rejects() {
    let mut case = clear_base_case();
    tamper_clear_eval(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_clear_opening_point_rejects() {
    let mut case = clear_base_case();
    tamper_opening_point(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_opening_point_rejects() {
    let mut case = zk_base_case();
    tamper_opening_point(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_checked_input_digest_rejects() {
    let mut case = clear_base_case();
    tamper_checked_input_digest(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_verifier_setup_digest_rejects() {
    let mut case = clear_base_case();
    tamper_verifier_setup_digest(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_verifier_setup_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_verifier_setup_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dory_proof_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_proof_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dory_vmv_c_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_vmv_c_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dory_vmv_e1_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_vmv_e1_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dory_zk_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_zk_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_multiround_dory_e2_artifact_rejects() {
    let mut case = zk_multiround_case();
    tamper_dory_zk_artifact(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_zk_multiround_dory_y_com_artifact_rejects() {
    let mut case = zk_multiround_case();
    tamper_dory_zk_y_com_artifact(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_zk_multiround_dory_scalar_product_artifact_rejects() {
    let mut case = zk_multiround_case();
    tamper_dory_scalar_product_artifact(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_dory_reduce_round_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_reduce_round_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_multiround_dory_reduce_round_artifact_rejects() {
    let mut case = zk_multiround_case();
    tamper_dory_reduce_round_artifact(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_dory_reduce_dimensions_rejects() {
    let mut case = clear_base_case();
    tamper_dory_reduce_dimensions(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_multiround_dory_reduce_dimensions_rejects() {
    let mut case = zk_multiround_case();
    tamper_dory_reduce_dimensions(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_gt_dimensions_rejects() {
    let mut case = clear_base_case();
    tamper_gt_dimensions(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_packing_dimensions_rejects() {
    let mut case = clear_base_case();
    tamper_packing_dimensions(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dory_final_artifact_rejects() {
    let mut case = clear_base_case();
    tamper_dory_final_artifact(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_jolt_commitment_claim_rejects() {
    let mut case = clear_base_case();
    tamper_jolt_commitment_claim(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_jolt_commitment_gt_claim_rejects() {
    let mut case = clear_base_case();
    tamper_jolt_commitment_gt_claim(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_jolt_evaluation_claim_rejects() {
    let mut case = clear_base_case();
    tamper_jolt_evaluation_claim(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_transcript_scalar_claim_rejects() {
    let mut case = clear_base_case();
    tamper_transcript_scalar_claim(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_multiround_sigma_c_transcript_scalar_claim_rejects() {
    let mut case = zk_multiround_case();
    tamper_zk_sigma_c_transcript_scalar_claim(&mut case);
    assert_rejects(case.verify_zk());
}
