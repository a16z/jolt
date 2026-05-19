use crate::support::{FixtureId, TestCase, VerifierCheckpoint};

pub const STANDARD_MULDIV_SMALL: TestCase = TestCase {
    name: "standard_muldiv_small",
    zk: false,
    fixture: FixtureId::MulDivSmall,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const STANDARD_FIBONACCI_SMALL: TestCase = TestCase {
    name: "standard_fibonacci_small",
    zk: false,
    fixture: FixtureId::FibonacciSmall,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const STANDARD_FIBONACCI_MEDIUM: TestCase = TestCase {
    name: "standard_fibonacci_medium",
    zk: false,
    fixture: FixtureId::FibonacciMedium,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const STANDARD_MEMORY_OPS: TestCase = TestCase {
    name: "standard_memory_ops",
    zk: false,
    fixture: FixtureId::MemoryOps,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const STANDARD_COLLATZ_SMALL: TestCase = TestCase {
    name: "standard_collatz_small",
    zk: false,
    fixture: FixtureId::CollatzSmall,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const STANDARD_SHA2_SMALL: TestCase = TestCase {
    name: "standard_sha2_small",
    zk: false,
    fixture: FixtureId::Sha2Small,
    first_checked_at: VerifierCheckpoint::Commitments,
};

pub const ZK_MULDIV_SMALL: TestCase = TestCase {
    name: "zk_muldiv_small",
    zk: true,
    fixture: FixtureId::MulDivZkSmall,
    first_checked_at: VerifierCheckpoint::Zk,
};

pub const ZK_STAGE1_PREFIX: TestCase = TestCase {
    name: "zk_stage1_prefix",
    zk: true,
    fixture: FixtureId::ZkStage1Prefix,
    first_checked_at: VerifierCheckpoint::Stage1,
};

pub const STANDARD_ADVICE_CONSUMER: TestCase = TestCase {
    name: "standard_advice_consumer",
    zk: false,
    fixture: FixtureId::AdviceConsumer,
    first_checked_at: VerifierCheckpoint::Commitments,
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
