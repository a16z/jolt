use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "direct commitment tampering fixtures are not wired yet"]
fn tampered_commitment_order_reject() {
    assert_eq!(
        soundness_expectation(tampering::COMMITMENT_ORDER),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
