use crate::support::{FixtureId, TestCase, VerifierPhase};

pub const STANDARD_MULDIV_SMALL: TestCase = TestCase {
    name: "standard_muldiv_small",
    zk: false,
    fixture: FixtureId::MulDivSmall,
    checked_at: VerifierPhase::Commitments,
};

pub const STANDARD_FIBONACCI_SMALL: TestCase = TestCase {
    name: "standard_fibonacci_small",
    zk: false,
    fixture: FixtureId::FibonacciSmall,
    checked_at: VerifierPhase::Commitments,
};

pub const STANDARD_FIBONACCI_MEDIUM: TestCase = TestCase {
    name: "standard_fibonacci_medium",
    zk: false,
    fixture: FixtureId::FibonacciMedium,
    checked_at: VerifierPhase::Commitments,
};

pub const STANDARD_MEMORY_OPS: TestCase = TestCase {
    name: "standard_memory_ops",
    zk: false,
    fixture: FixtureId::MemoryOps,
    checked_at: VerifierPhase::Commitments,
};

pub const STANDARD_COLLATZ_SMALL: TestCase = TestCase {
    name: "standard_collatz_small",
    zk: false,
    fixture: FixtureId::CollatzSmall,
    checked_at: VerifierPhase::Commitments,
};

pub const STANDARD_SHA2_SMALL: TestCase = TestCase {
    name: "standard_sha2_small",
    zk: false,
    fixture: FixtureId::Sha2Small,
    checked_at: VerifierPhase::Commitments,
};

pub const ZK_MULDIV_SMALL: TestCase = TestCase {
    name: "zk_muldiv_small",
    zk: true,
    fixture: FixtureId::MulDivZkSmall,
    checked_at: VerifierPhase::Zk,
};

pub const ZK_STAGE1_PREFIX: TestCase = TestCase {
    name: "zk_stage1_prefix",
    zk: true,
    fixture: FixtureId::ZkStage1Prefix,
    checked_at: VerifierPhase::Stage1,
};

pub const STANDARD_ADVICE_CONSUMER: TestCase = TestCase {
    name: "standard_advice_consumer",
    zk: false,
    fixture: FixtureId::AdviceConsumer,
    checked_at: VerifierPhase::Commitments,
};

pub const ALL: &[TestCase] = &[
    STANDARD_MULDIV_SMALL,
    STANDARD_FIBONACCI_SMALL,
    STANDARD_FIBONACCI_MEDIUM,
    STANDARD_MEMORY_OPS,
    STANDARD_COLLATZ_SMALL,
    STANDARD_SHA2_SMALL,
    ZK_MULDIV_SMALL,
    ZK_STAGE1_PREFIX,
    STANDARD_ADVICE_CONSUMER,
];
