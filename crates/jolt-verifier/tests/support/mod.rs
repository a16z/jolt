#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerifierCheckpoint {
    Preamble,
    Commitments,
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Stage8Openings,
    Zk,
    Full,
}

pub const CURRENT_VERIFIER_FRONTIER: VerifierCheckpoint = VerifierCheckpoint::Commitments;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureId {
    MulDivSmall,
    MulDivZkSmall,
    AdviceCommitments,
    PublicIoMismatch,
    TrustedAdviceMismatch,
    MixedProofMode,
    ConfigMismatch,
    OpeningClaimMismatch,
    BlindFoldMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestCase {
    pub name: &'static str,
    pub zk: bool,
    pub fixture: FixtureId,
    pub first_checked_at: VerifierCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarnessExpectation {
    ReachesFrontier,
    RejectsAtOrBeforeFrontier,
    FutureCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixtureMetadata {
    pub id: FixtureId,
    pub name: &'static str,
    pub zk: bool,
    pub has_trusted_advice: bool,
    pub expected_core_accepts: bool,
    pub notes: &'static str,
}

pub fn completeness_expectation(case: TestCase) -> HarnessExpectation {
    if case.first_checked_at <= CURRENT_VERIFIER_FRONTIER {
        HarnessExpectation::ReachesFrontier
    } else {
        HarnessExpectation::FutureCheckpoint
    }
}

pub fn soundness_expectation(case: TestCase) -> HarnessExpectation {
    if case.first_checked_at <= CURRENT_VERIFIER_FRONTIER {
        HarnessExpectation::RejectsAtOrBeforeFrontier
    } else {
        HarnessExpectation::FutureCheckpoint
    }
}

pub fn assert_unique_case_names(cases: &[TestCase]) {
    for (index, case) in cases.iter().enumerate() {
        for other in &cases[index + 1..] {
            assert_ne!(case.name, other.name, "duplicate test case name");
        }
    }
}

pub fn assert_case_metadata_matches(case: TestCase, metadata: FixtureMetadata) {
    assert_eq!(case.fixture, metadata.id);
    assert_eq!(case.zk, metadata.zk);
}

pub fn assert_reaches_current_frontier(result: Result<(), VerifierError>) {
    let reached_frontier = match result {
        Ok(()) => CURRENT_VERIFIER_FRONTIER == VerifierCheckpoint::Full,
        Err(VerifierError::Unimplemented) => CURRENT_VERIFIER_FRONTIER < VerifierCheckpoint::Full,
        Err(_) => false,
    };

    assert!(
        reached_frontier,
        "valid proof did not reach current verifier frontier"
    );
}

pub fn assert_rejects_at_or_before_current_frontier(result: Result<(), VerifierError>) {
    let rejected = match result {
        Ok(()) | Err(VerifierError::Unimplemented) => false,
        Err(_) => true,
    };

    assert!(
        rejected,
        "tampered proof was accepted or reached unimplemented verifier code"
    );
}
#[cfg(feature = "core-fixtures")]
pub mod core_fixtures;
pub mod dory_pedersen;

use jolt_verifier::VerifierError;
