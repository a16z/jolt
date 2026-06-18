pub mod commitments;
pub mod configs;
pub mod openings;
pub mod preamble;
pub mod proof_shape;
pub mod zk;

use crate::support::{FixtureId, TestCase, VerifierPhase};

pub const PUBLIC_IO_MISMATCH: TestCase = TestCase {
    name: "prover_transitivity_public_io_mismatch",
    zk: false,
    fixture: FixtureId::PublicIoMismatch,
    checked_at: VerifierPhase::Preamble,
};

pub const TRUSTED_ADVICE_MISMATCH: TestCase = TestCase {
    name: "prover_transitivity_trusted_advice_mismatch",
    zk: false,
    fixture: FixtureId::TrustedAdviceMismatch,
    checked_at: VerifierPhase::Stage1,
};

pub const MIXED_PROOF_MODE: TestCase = TestCase {
    name: "prover_transitivity_mixed_proof_mode",
    zk: false,
    fixture: FixtureId::MixedProofMode,
    checked_at: VerifierPhase::Preamble,
};

pub const CONFIG_MISMATCH: TestCase = TestCase {
    name: "prover_transitivity_config_mismatch",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    checked_at: VerifierPhase::Preamble,
};

pub const OPENING_CLAIM_MISMATCH: TestCase = TestCase {
    name: "prover_transitivity_opening_claim_mismatch",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    checked_at: VerifierPhase::Stage8Openings,
};

pub const BLINDFOLD_MISMATCH: TestCase = TestCase {
    name: "prover_transitivity_blindfold_mismatch",
    zk: true,
    fixture: FixtureId::BlindFoldMismatch,
    checked_at: VerifierPhase::Zk,
};

pub const ALL: &[TestCase] = &[
    PUBLIC_IO_MISMATCH,
    TRUSTED_ADVICE_MISMATCH,
    MIXED_PROOF_MODE,
    CONFIG_MISMATCH,
    OPENING_CLAIM_MISMATCH,
    BLINDFOLD_MISMATCH,
];
