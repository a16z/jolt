#![cfg_attr(
    all(feature = "core-fixtures", feature = "zk"),
    expect(
        clippy::panic,
        reason = "ZK tampering helpers assert fixture shape before mutating real core proofs"
    )
)]

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support;
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn missing_zk_vector_commitment_setup_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    case.preprocessing.vc_setup = None;

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage1_remainder_round_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_round(&mut case.proof.stages.stage1_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_uniskip_round_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_batch_output_commitment_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_output_claim_row(&mut case.proof.stages.stage2_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_round(&mut case.proof.stages.stage3_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_degree_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage3_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_output_commitment_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_output_claim_row(&mut case.proof.stages.stage3_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_round(&mut case.proof.stages.stage4_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_degree_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage4_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_output_commitment_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_output_claim_row(&mut case.proof.stages.stage4_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_round(&mut case.proof.stages.stage5_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_degree_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage5_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_output_commitment_count_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    pop_committed_output_claim_row(&mut case.proof.stages.stage5_sumcheck_proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn pop_committed_round<F, C>(proof: &mut jolt_sumcheck::SumcheckProof<F, C>)
where
    F: jolt_field::Field,
{
    let jolt_sumcheck::SumcheckProof::Committed(proof) = proof else {
        panic!("ZK fixture must use committed sumcheck proofs");
    };
    let _ = proof.rounds.pop();
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn exceed_first_committed_round_degree_bound<F, C>(proof: &mut jolt_sumcheck::SumcheckProof<F, C>)
where
    F: jolt_field::Field,
{
    let jolt_sumcheck::SumcheckProof::Committed(proof) = proof else {
        panic!("ZK fixture must use committed sumcheck proofs");
    };
    let Some(round) = proof.rounds.first_mut() else {
        panic!("ZK committed sumcheck proof must have at least one round");
    };
    round.degree = usize::MAX;
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn pop_committed_output_claim_row<F, C>(proof: &mut jolt_sumcheck::SumcheckProof<F, C>)
where
    F: jolt_field::Field,
{
    let jolt_sumcheck::SumcheckProof::Committed(proof) = proof else {
        panic!("ZK fixture must use committed sumcheck proofs");
    };
    let _ = proof.output_claims.commitments.pop();
}

#[cfg(any(not(feature = "core-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features core-fixtures,zk to live-generate, cast, and tamper real core ZK proofs"]
fn missing_zk_vector_commitment_setup_rejects_now() {
    assert_eq!(
        soundness_expectation(tampering::BLINDFOLD_PROOF),
        HarnessExpectation::FutureCheckpoint,
    );
}

#[test]
#[ignore = "BlindFold verification is not wired yet"]
fn tampered_blindfold_proof_reject() {
    assert_eq!(
        soundness_expectation(tampering::BLINDFOLD_PROOF),
        HarnessExpectation::FutureCheckpoint,
    );
}
