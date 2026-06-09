use crate::support::{FixtureId, TestCase, VerifierPhase};

pub const CLEAR_BASE: TestCase = TestCase {
    name: "clear_valid_assist_proof_accepts",
    zk: false,
    fixture: FixtureId::ClearBase,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ZK_BASE: TestCase = TestCase {
    name: "zk_valid_assist_proof_accepts",
    zk: true,
    fixture: FixtureId::ZkBase,
    checked_at: VerifierPhase::NativeOutput,
};

pub const CLEAR_MULTIROUND: TestCase = TestCase {
    name: "clear_multiround_valid_assist_proof_accepts",
    zk: false,
    fixture: FixtureId::ClearMultiround,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ZK_MULTIROUND: TestCase = TestCase {
    name: "zk_multiround_valid_assist_proof_accepts",
    zk: true,
    fixture: FixtureId::ZkMultiround,
    checked_at: VerifierPhase::NativeOutput,
};

pub const ALL: &[TestCase] = &[CLEAR_BASE, ZK_BASE, CLEAR_MULTIROUND, ZK_MULTIROUND];
