#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        reason = "ZK tampering helpers assert fixture shape before mutating real verifier objects"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::tamper_manifest;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn missing_zk_vector_commitment_setup_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        case.preprocessing.vc_setup = None;

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage1_remainder_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 2, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_uniskip_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 4, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage2_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 5, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 8, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 9, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage3_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 10, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 12, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 13, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage4_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 14, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 16, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 17, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage5_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 18, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_address_phase_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 20, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_address_phase_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 21, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_address_phase_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 22, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_cycle_phase_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 24, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_cycle_phase_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 25, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage6_cycle_phase_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 26, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_round_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 28, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_round_degree_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 29, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_stage7_batch_output_commitment_count_rejects_now() {
    with_zk_verifier_stack(|| {
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 30, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_zk_joint_opening_eval_commitment_rejects_now() {
    with_zk_verifier_stack(|| {
        assert_zk_target_active("zk.joint_opening_proof.eval_commitment");
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        case.proof.joint_opening_proof.0.y_com = None;

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn tampered_blindfold_proof_rejects_now() {
    with_zk_verifier_stack(|| {
        assert_zk_target_active("zk.blindfold_proof");
        let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
        tamper_narg_at(&mut case.proof.narg, 31, 32);

        support::assert_zk_rejects(case.verify());
    });
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn with_zk_verifier_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("zk-verifier-tamper".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(test)
        .expect("spawn ZK verifier tamper test")
        .join()
        .expect("ZK verifier tamper test panicked");
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn tamper_narg_at(narg: &mut [u8], numerator: usize, denominator: usize) {
    assert!(!narg.is_empty(), "ZK fixture must carry NARG proof data");
    assert!(denominator > 0, "denominator must be non-zero");
    let scaled = narg.len().saturating_mul(numerator) / denominator;
    let index = scaled.min(narg.len() - 1);
    narg[index] ^= 1;
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn assert_zk_target_active(name: &str) {
    let target = tamper_manifest::required_target(name);
    tamper_manifest::assert_manifest_target_is_active(target);
    assert!(
        target.mode.includes(true),
        "tamper target mode does not include ZK: {target:?}"
    );
}

#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features prover-fixtures,zk to live-generate and tamper verifier-native ZK proofs"]
fn missing_zk_vector_commitment_setup_rejects_now() {}

#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[test]
#[ignore = "enable --features prover-fixtures,zk to live-generate and tamper verifier-native ZK proofs"]
fn tampered_blindfold_proof_reject() {}
