use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "final output claim checks are not wired yet"]
fn tampered_output_claim_reject() {
    assert_eq!(
        soundness_expectation(tampering::OUTPUT_CLAIM),
        HarnessExpectation::FutureCheckpoint,
    );
}
