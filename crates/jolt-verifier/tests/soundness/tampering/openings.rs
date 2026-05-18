use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "opening verification is not wired yet"]
fn tampered_opening_value_reject() {
    assert_eq!(
        soundness_expectation(tampering::OPENING_VALUE),
        HarnessExpectation::FutureCheckpoint,
    );
}
