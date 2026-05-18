use crate::{
    soundness::core_transitivity,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "ZK invalid-proof fixture generation is not wired yet"]
fn core_rejects_blindfold_mismatch() {
    assert_eq!(
        soundness_expectation(core_transitivity::BLINDFOLD_MISMATCH),
        HarnessExpectation::FutureCheckpoint,
    );
}
