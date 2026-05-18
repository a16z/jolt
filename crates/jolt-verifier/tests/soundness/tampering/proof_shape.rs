use crate::{
    soundness::tampering,
    support,
    support::dory_pedersen,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
fn mixed_clear_and_committed_stage_proofs_reject_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.stages.stage5_sumcheck_proof =
        jolt_sumcheck::SumcheckProof::Committed(jolt_sumcheck::CommittedSumcheckProof::default());

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn mixed_uniskip_stage_proof_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.stages.stage1_uni_skip_first_round_proof =
        jolt_sumcheck::SumcheckProof::Committed(jolt_sumcheck::CommittedSumcheckProof::default());

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn missing_standard_opening_claims_reject_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.opening_claims = None;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn unexpected_standard_blindfold_proof_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.blindfold_proof = Some(());

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn missing_zk_blindfold_proof_rejects_now() {
    let mut case = dory_pedersen::zk_case();
    case.proof.blindfold_proof = None;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn unexpected_zk_opening_claims_reject_now() {
    let mut case = dory_pedersen::zk_case();
    case.proof.opening_claims = Some(());

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn clear_stage_in_zk_proof_rejects_now() {
    let mut case = dory_pedersen::zk_case();
    case.proof.stages.stage3_sumcheck_proof = jolt_sumcheck::SumcheckProof::Clear(
        jolt_sumcheck::ClearProof::Compressed(jolt_sumcheck::CompressedSumcheckProof::default()),
    );

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
#[ignore = "direct proof-shape tampering fixtures are not wired yet"]
fn tampered_mixed_proof_shape_reject() {
    assert_eq!(
        soundness_expectation(tampering::MIXED_PROOF_SHAPE),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
