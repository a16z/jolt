use crate::support::{FixtureId, TestCase, VerifierCheckpoint};

pub const STANDARD_MULDIV_SMALL: TestCase = TestCase {
    name: "standard_muldiv_small",
    zk: false,
    fixture: FixtureId::MulDivSmall,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const ZK_MULDIV_SMALL: TestCase = TestCase {
    name: "zk_muldiv_small",
    zk: true,
    fixture: FixtureId::MulDivZkSmall,
    first_checked_at: VerifierCheckpoint::Zk,
};

pub const STANDARD_ADVICE_COMMITMENTS: TestCase = TestCase {
    name: "standard_advice_commitments",
    zk: false,
    fixture: FixtureId::AdviceCommitments,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const ALL: &[TestCase] = &[
    STANDARD_MULDIV_SMALL,
    ZK_MULDIV_SMALL,
    STANDARD_ADVICE_COMMITMENTS,
];
