#![cfg_attr(
    feature = "prover-fixtures",
    expect(
        clippy::panic,
        reason = "fixture helpers should fail loudly when stored verifier NARG is malformed"
    )
)]

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
    CommittedMulDivSmall,
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
    pub expected_prover_accepts: bool,
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

#[cfg(feature = "prover-fixtures")]
#[derive(Clone)]
pub struct NargFrameRange {
    pub full: std::ops::Range<usize>,
    pub body: std::ops::Range<usize>,
}

#[cfg(feature = "prover-fixtures")]
pub fn narg_frame_ranges(narg: &[u8]) -> Vec<NargFrameRange> {
    let mut offset = 0;
    let mut ranges = Vec::new();
    while offset < narg.len() {
        assert!(
            narg.len() - offset >= 8,
            "NARG has a truncated frame prefix at byte {offset}"
        );
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&narg[offset..offset + 8]);
        let len_u64 = u64::from_le_bytes(len_bytes);
        assert!(
            usize::BITS >= 64 || len_u64 <= usize::MAX as u64,
            "NARG frame length should fit in usize"
        );
        let len = len_u64 as usize;
        let body_start = offset + 8;
        let body_end = body_start
            .checked_add(len)
            .unwrap_or_else(|| panic!("NARG frame body range should not overflow"));
        assert!(
            body_end <= narg.len(),
            "NARG frame at byte {offset} is truncated"
        );
        ranges.push(NargFrameRange {
            full: offset..body_end,
            body: body_start..body_end,
        });
        offset = body_end;
    }
    ranges
}

#[cfg(feature = "prover-fixtures")]
pub fn narg_frame_has_body(narg: &[u8], frame_index: usize) -> bool {
    narg_frame_ranges(narg)
        .get(frame_index)
        .is_some_and(|range| !range.body.is_empty())
}

#[cfg(feature = "prover-fixtures")]
pub fn replace_narg_frame_body(narg: &mut Vec<u8>, frame_index: usize, body: Vec<u8>) {
    let ranges = narg_frame_ranges(narg);
    let Some(range) = ranges.get(frame_index) else {
        panic!("NARG is missing expected frame {frame_index}");
    };
    let len_bytes = (body.len() as u64).to_le_bytes();
    narg[range.full.start..range.full.start + 8].copy_from_slice(&len_bytes);
    drop(narg.splice(range.body.clone(), body));
}
#[cfg(feature = "prover-fixtures")]
pub mod proof_claims;
pub mod tamper_manifest;
#[cfg(feature = "prover-fixtures")]
pub mod verifier_fixtures;
#[cfg(feature = "zk")]
pub mod zk_audit;

use jolt_verifier::VerifierError;
