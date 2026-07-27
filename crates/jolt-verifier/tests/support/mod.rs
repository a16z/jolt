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
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
pub mod akita_fixtures;
#[cfg(feature = "prover-fixtures")]
pub mod proof_claims;
pub mod tamper_manifest;
#[cfg(all(feature = "prover-fixtures", not(feature = "akita")))]
pub mod verifier_fixtures;
#[cfg(feature = "zk")]
pub mod zk_audit;

use jolt_verifier::VerifierError;
