#![cfg_attr(
    all(feature = "core-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "ZK tampering helpers assert fixture shape before mutating real core proofs"
    )
)]

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support::tamper_manifest;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_field::FromPrimitiveInt as _;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn missing_zk_vector_commitment_setup_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        case.preprocessing.vc_setup = None;

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage1_remainder_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage1_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_uniskip_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage2_uni_skip_first_round_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage2_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage3_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage3_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage3_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage4_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage4_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage4_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage5_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage5_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage5_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage6_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage6_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage6_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_round(&mut case.proof.stages.stage7_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        exceed_first_committed_round_degree_bound(&mut case.proof.stages.stage7_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        pop_committed_output_claim_row(&mut case.proof.stages.stage7_sumcheck_proof);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_joint_opening_eval_commitment_rejects_now() {
    with_zk_verifier_stack(|| {
        assert_zk_target_active("zk.joint_opening_proof.eval_commitment");
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        case.proof.joint_opening_proof.0.y_com = None;

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
fn tampered_blindfold_proof_rejects_now() {
    with_zk_verifier_stack(|| {
        assert_zk_target_active("zk.blindfold_proof");
        let mut case = crate::support::core_fixtures::zk_muldiv_case();
        let JoltProofClaims::Zk { blindfold_proof } = &mut case.proof.claims else {
            panic!("ZK fixture must carry a BlindFold proof");
        };
        blindfold_proof.random_u += jolt_field::Fr::from_u64(1);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn with_zk_verifier_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("zk-verifier-tamper".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(test)
        .expect("spawn ZK verifier tamper test")
        .join()
        .expect("ZK verifier tamper test panicked");
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
    round.degree = round.degree.saturating_add(1024);
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

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn assert_zk_target_active(name: &str) {
    let target = tamper_manifest::required_target(name);
    tamper_manifest::assert_manifest_target_is_active(target);
    assert!(
        target.mode.includes(true),
        "tamper target mode does not include ZK: {target:?}"
    );
}

#[cfg(any(not(feature = "core-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features core-fixtures,zk to live-generate, cast, and tamper real core ZK proofs"]
fn missing_zk_vector_commitment_setup_rejects_now() {}

#[cfg(any(not(feature = "core-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features core-fixtures,zk to live-generate, cast, and tamper real core ZK proofs"]
fn tampered_blindfold_proof_reject() {}
