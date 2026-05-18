use crate::{
    soundness::core_transitivity,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "opening-claim rejection fixtures are not wired yet"]
fn core_rejects_opening_claim_mismatch() {
    assert_eq!(
        soundness_expectation(core_transitivity::OPENING_CLAIM_MISMATCH),
        HarnessExpectation::FutureCheckpoint,
    );
}
