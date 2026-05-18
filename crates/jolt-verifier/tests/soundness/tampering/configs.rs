use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "direct config tampering fixtures are not wired yet"]
fn tampered_trace_length_reject() {
    assert_eq!(
        soundness_expectation(tampering::CONFIG_TRACE_LENGTH),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
