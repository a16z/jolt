pub mod commitments;
pub mod configs;
pub mod openings;
pub mod preamble;
pub mod proof_shape;
pub mod zk;

use crate::support::{FixtureId, TestCase, VerifierCheckpoint};

pub const PUBLIC_IO_MISMATCH: TestCase = TestCase {
    name: "core_transitivity_public_io_mismatch",
    zk: false,
    fixture: FixtureId::PublicIoMismatch,
    first_checked_at: VerifierCheckpoint::Preamble,
};

pub const TRUSTED_ADVICE_MISMATCH: TestCase = TestCase {
    name: "core_transitivity_trusted_advice_mismatch",
    zk: false,
    fixture: FixtureId::TrustedAdviceMismatch,
    first_checked_at: VerifierCheckpoint::Stage1,
};

pub const MIXED_PROOF_MODE: TestCase = TestCase {
    name: "core_transitivity_mixed_proof_mode",
    zk: false,
    fixture: FixtureId::MixedProofMode,
    first_checked_at: VerifierCheckpoint::Preamble,
};

pub const CONFIG_MISMATCH: TestCase = TestCase {
    name: "core_transitivity_config_mismatch",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Preamble,
};

pub const OPENING_CLAIM_MISMATCH: TestCase = TestCase {
    name: "core_transitivity_opening_claim_mismatch",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    first_checked_at: VerifierCheckpoint::Stage8Openings,
};

pub const BLINDFOLD_MISMATCH: TestCase = TestCase {
    name: "core_transitivity_blindfold_mismatch",
    zk: true,
    fixture: FixtureId::BlindFoldMismatch,
    first_checked_at: VerifierCheckpoint::Zk,
};

pub const ALL: &[TestCase] = &[
    PUBLIC_IO_MISMATCH,
    TRUSTED_ADVICE_MISMATCH,
    MIXED_PROOF_MODE,
    CONFIG_MISMATCH,
    OPENING_CLAIM_MISMATCH,
    BLINDFOLD_MISMATCH,
];
