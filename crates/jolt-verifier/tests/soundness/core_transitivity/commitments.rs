use crate::{
    soundness::core_transitivity,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "core invalid-proof fixture generation is not wired yet"]
fn core_rejects_trusted_advice_commitment_mismatch() {
    assert_eq!(
        soundness_expectation(core_transitivity::TRUSTED_ADVICE_MISMATCH),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
