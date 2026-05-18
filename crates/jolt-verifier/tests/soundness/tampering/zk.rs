use crate::{
    soundness::tampering,
    support,
    support::dory_pedersen,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
fn missing_zk_vector_commitment_setup_rejects_now() {
    let mut case = dory_pedersen::zk_case();
    case.preprocessing.vc_setup = None;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
#[ignore = "BlindFold verification is not wired yet"]
fn tampered_blindfold_proof_reject() {
    assert_eq!(
        soundness_expectation(tampering::BLINDFOLD_PROOF),
        HarnessExpectation::FutureCheckpoint,
    );
}
