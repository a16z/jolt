#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerifierPhase {
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
}

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
    pub checked_at: VerifierPhase,
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

pub fn assert_accepts_mode(zk: bool, result: Result<(), VerifierError>) {
    let result_debug = format!("{result:?}");

    assert!(
        result.is_ok(),
        "valid {} proof was rejected: {result_debug}",
        if zk { "ZK" } else { "standard" }
    );
}

pub fn assert_accepts(result: Result<(), VerifierError>) {
    assert_accepts_mode(false, result);
}

pub fn assert_zk_accepts(result: Result<(), VerifierError>) {
    assert_accepts_mode(true, result);
}

pub fn assert_rejects_mode(zk: bool, result: Result<(), VerifierError>) {
    let result_debug = format!("{result:?}");
    let rejected = match result {
        Ok(()) => false,
        Err(_) => true,
    };

    assert!(
        rejected,
        "tampered {} proof was accepted or reached unimplemented verifier code: {result_debug}",
        if zk { "ZK" } else { "standard" }
    );
}

pub fn assert_rejects(result: Result<(), VerifierError>) {
    assert_rejects_mode(false, result);
}

pub fn assert_zk_rejects(result: Result<(), VerifierError>) {
    assert_rejects_mode(true, result);
}
#[cfg(feature = "core-fixtures")]
pub mod core_fixtures;
pub mod tamper_manifest;

use jolt_verifier::VerifierError;
