pub mod commitments;
pub mod configs;
pub mod manifest;
pub mod openings;
pub mod output_claims;
pub mod preamble;
pub mod proof_shape;
pub mod sumcheck;
pub mod zk;

use crate::support::{FixtureId, TestCase, VerifierCheckpoint};

pub const PUBLIC_INPUT_BYTES: TestCase = TestCase {
    name: "tamper_public_input_bytes",
    zk: false,
    fixture: FixtureId::PublicIoMismatch,
    first_checked_at: VerifierCheckpoint::Stage1,
};

pub const COMMITMENT_ORDER: TestCase = TestCase {
    name: "tamper_commitment_order",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage1,
};

pub const MIXED_PROOF_SHAPE: TestCase = TestCase {
    name: "tamper_mixed_proof_shape",
    zk: false,
    fixture: FixtureId::MixedProofMode,
    first_checked_at: VerifierCheckpoint::Preamble,
};

pub const CONFIG_TRACE_LENGTH: TestCase = TestCase {
    name: "tamper_trace_length",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Preamble,
};

pub const STAGE1_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage1_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage1,
};

pub const STAGE2_UNISKIP_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage2_uniskip_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage2,
};

pub const STAGE2_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage2_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage2,
};

pub const STAGE3_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage3_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage3,
};

pub const STAGE4_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage4_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage4,
};

pub const STAGE5_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage5_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage5,
};

pub const STAGE6_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage6_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage6,
};

pub const STAGE7_SUMCHECK_PAYLOAD: TestCase = TestCase {
    name: "tamper_stage7_sumcheck_payload",
    zk: false,
    fixture: FixtureId::ConfigMismatch,
    first_checked_at: VerifierCheckpoint::Stage7,
};

pub const OPENING_VALUE: TestCase = TestCase {
    name: "tamper_opening_value",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    first_checked_at: VerifierCheckpoint::Stage8Openings,
};

pub const OUTPUT_CLAIM: TestCase = TestCase {
    name: "tamper_output_claim",
    zk: false,
    fixture: FixtureId::OpeningClaimMismatch,
    first_checked_at: VerifierCheckpoint::Stage8Openings,
};

pub const BLINDFOLD_PROOF: TestCase = TestCase {
    name: "tamper_blindfold_proof",
    zk: true,
    fixture: FixtureId::BlindFoldMismatch,
    first_checked_at: VerifierCheckpoint::Zk,
};

pub const ALL: &[TestCase] = &[
    PUBLIC_INPUT_BYTES,
    COMMITMENT_ORDER,
    MIXED_PROOF_SHAPE,
    CONFIG_TRACE_LENGTH,
    STAGE1_SUMCHECK_PAYLOAD,
    STAGE2_UNISKIP_PAYLOAD,
    STAGE2_SUMCHECK_PAYLOAD,
    STAGE3_SUMCHECK_PAYLOAD,
    STAGE4_SUMCHECK_PAYLOAD,
    STAGE5_SUMCHECK_PAYLOAD,
    STAGE6_SUMCHECK_PAYLOAD,
    STAGE7_SUMCHECK_PAYLOAD,
    OPENING_VALUE,
    OUTPUT_CLAIM,
    BLINDFOLD_PROOF,
];
