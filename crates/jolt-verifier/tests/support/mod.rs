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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierFrontiers {
    pub standard: VerifierCheckpoint,
    pub zk: VerifierCheckpoint,
}

impl VerifierFrontiers {
    pub const fn for_mode(self, zk: bool) -> VerifierCheckpoint {
        if zk {
            self.zk
        } else {
            self.standard
        }
    }
}

pub const CURRENT_VERIFIER_FRONTIERS: VerifierFrontiers = VerifierFrontiers {
    standard: VerifierCheckpoint::Full,
    zk: VerifierCheckpoint::Stage8Openings,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureId {
    MulDivSmall,
    FibonacciSmall,
    FibonacciMedium,
    MemoryOps,
    CollatzSmall,
    Sha2Small,
    MulDivZkSmall,
    ZkStage1Prefix,
    AdviceConsumer,
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

pub fn current_verifier_frontier(zk: bool) -> VerifierCheckpoint {
    CURRENT_VERIFIER_FRONTIERS.for_mode(zk)
}

pub fn completeness_expectation(case: TestCase) -> HarnessExpectation {
    if case.first_checked_at <= current_verifier_frontier(case.zk) {
        HarnessExpectation::ReachesFrontier
    } else {
        HarnessExpectation::FutureCheckpoint
    }
}

pub fn soundness_expectation(case: TestCase) -> HarnessExpectation {
    if case.first_checked_at <= current_verifier_frontier(case.zk) {
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

pub fn assert_reaches_frontier(zk: bool, result: Result<(), VerifierError>) {
    let frontier = current_verifier_frontier(zk);
    let result_debug = format!("{result:?}");
    let reached_frontier = match result {
        Ok(()) => frontier == VerifierCheckpoint::Full,
        Err(VerifierError::Unimplemented) => frontier < VerifierCheckpoint::Full,
        Err(_) => false,
    };

    assert!(
        reached_frontier,
        "valid {} proof did not reach current verifier frontier {frontier:?}: {result_debug}",
        if zk { "ZK" } else { "standard" }
    );
}

pub fn assert_reaches_current_frontier(result: Result<(), VerifierError>) {
    assert_reaches_frontier(false, result);
}

pub fn assert_zk_reaches_current_frontier(result: Result<(), VerifierError>) {
    assert_reaches_frontier(true, result);
}

pub fn assert_rejects_at_or_before_frontier(zk: bool, result: Result<(), VerifierError>) {
    let frontier = current_verifier_frontier(zk);
    let result_debug = format!("{result:?}");
    let rejected = match result {
        Ok(()) | Err(VerifierError::Unimplemented) => false,
        Err(_) => true,
    };

    assert!(
        rejected,
        "tampered {} proof was accepted or reached unimplemented verifier code before frontier {frontier:?}: {result_debug}",
        if zk { "ZK" } else { "standard" }
    );
}

pub fn assert_rejects_at_or_before_current_frontier(result: Result<(), VerifierError>) {
    assert_rejects_at_or_before_frontier(false, result);
}

pub fn assert_zk_rejects_at_or_before_current_frontier(result: Result<(), VerifierError>) {
    assert_rejects_at_or_before_frontier(true, result);
}
#[cfg(feature = "core-fixtures")]
pub mod core_fixtures;
pub mod tamper_manifest;

use jolt_verifier::VerifierError;
