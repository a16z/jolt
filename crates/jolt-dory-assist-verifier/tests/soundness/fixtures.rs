use crate::support::{FixtureId, FixtureMetadata};

pub fn metadata(id: FixtureId) -> FixtureMetadata {
    match id {
        FixtureId::ClearInputMismatch => FixtureMetadata {
            id,
            name: "clear opening input mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered clear eval or opening point must be rejected.",
        },
        FixtureId::ZkInputMismatch => FixtureMetadata {
            id,
            name: "ZK opening input mismatch",
            zk: true,
            expected_accepts: false,
            notes: "Tampered ZK opening inputs, transcript scalars, and Dory proof artifacts must be rejected.",
        },
        FixtureId::StagePayloadMismatch => FixtureMetadata {
            id,
            name: "stage payload mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered stage payloads must fail before the final verifier result.",
        },
        FixtureId::OpeningClaimMismatch => FixtureMetadata {
            id,
            name: "packed opening claim mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered packed opening claims must be rejected by the Hyrax opening stage.",
        },
        FixtureId::HyraxOpeningMismatch => FixtureMetadata {
            id,
            name: "Hyrax opening proof mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered Hyrax opening proof payloads must be rejected.",
        },
        FixtureId::DenseCommitmentMismatch => FixtureMetadata {
            id,
            name: "dense witness commitment mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered packed witness commitments must be rejected.",
        },
        FixtureId::PublicOutputMismatch => FixtureMetadata {
            id,
            name: "public output mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered pre-final-exponentiation output must be rejected.",
        },
        FixtureId::ZkPublicOutputMismatch => FixtureMetadata {
            id,
            name: "ZK public output mismatch",
            zk: true,
            expected_accepts: false,
            notes: "Tampered ZK pre-final-exponentiation output must be rejected.",
        },
        FixtureId::NativeFinalInputMismatch => FixtureMetadata {
            id,
            name: "native-final input mismatch",
            zk: false,
            expected_accepts: false,
            notes: "Tampered native-final reducer-state input claim must be rejected.",
        },
        FixtureId::ZkNativeFinalInputMismatch => FixtureMetadata {
            id,
            name: "ZK native-final input mismatch",
            zk: true,
            expected_accepts: false,
            notes: "Tampered ZK native-final reducer-state input claim must be rejected.",
        },
        FixtureId::ClearBase
        | FixtureId::ClearMultiround
        | FixtureId::ZkBase
        | FixtureId::ZkMultiround => FixtureMetadata {
            id,
            name: "completeness-only fixture",
            zk: matches!(id, FixtureId::ZkBase | FixtureId::ZkMultiround),
            expected_accepts: true,
            notes: "Reserved for Dory-assist completeness tests.",
        },
    }
}
