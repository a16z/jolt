use crate::{
    soundness::core_transitivity,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[ignore = "core invalid-proof fixture generation is not wired yet"]
fn core_rejects_mixed_proof_mode() {
    assert_eq!(
        soundness_expectation(core_transitivity::MIXED_PROOF_MODE),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
