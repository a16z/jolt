use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "stage sumcheck verification is not wired yet"]
fn tampered_stage1_sumcheck_payload_reject() {
    assert_eq!(
        soundness_expectation(tampering::STAGE1_SUMCHECK_PAYLOAD),
        HarnessExpectation::FutureCheckpoint,
    );
}
