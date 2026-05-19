#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support;
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
#[ignore = "real ZK core fixture tampering is deferred until the ZK verifier frontier"]
fn missing_zk_vector_commitment_setup_rejects_now() {
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    case.preprocessing.vc_setup = None;

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
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
