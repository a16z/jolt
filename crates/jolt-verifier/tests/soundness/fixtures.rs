use crate::support::{FixtureId, FixtureMetadata};

pub fn metadata(id: FixtureId) -> FixtureMetadata {
    match id {
        FixtureId::PublicIoMismatch => FixtureMetadata {
            id,
            name: "public I/O mismatch",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "Core-transitivity target for preamble and public I/O binding.",
        },
        FixtureId::TrustedAdviceMismatch => FixtureMetadata {
            id,
            name: "trusted advice commitment mismatch",
            zk: false,
            has_trusted_advice: true,
            expected_core_accepts: false,
            notes: "Core-transitivity target for trusted advice commitment binding.",
        },
        FixtureId::MixedProofMode => FixtureMetadata {
            id,
            name: "mixed clear and committed proof mode",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "Proof-shape target for rejecting mixed sumcheck representations.",
        },
        FixtureId::ConfigMismatch => FixtureMetadata {
            id,
            name: "proof config mismatch",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "Config tampering target for read-write and one-hot settings.",
        },
        FixtureId::OpeningClaimMismatch => FixtureMetadata {
            id,
            name: "opening claim mismatch",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "Opening and final-claim mismatch target.",
        },
        FixtureId::BlindFoldMismatch => FixtureMetadata {
            id,
            name: "BlindFold mismatch",
            zk: true,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "ZK soundness target for BlindFold public inputs and proof payload.",
        },
        FixtureId::MulDivSmall
        | FixtureId::MulDivZkSmall
        | FixtureId::ZkStage1Prefix
        | FixtureId::AdviceCommitments => FixtureMetadata {
            id,
            name: "completeness-only fixture",
            zk: matches!(id, FixtureId::MulDivZkSmall | FixtureId::ZkStage1Prefix),
            has_trusted_advice: matches!(id, FixtureId::AdviceCommitments),
            expected_core_accepts: !matches!(id, FixtureId::ZkStage1Prefix),
            notes: "Reserved for completeness tests.",
        },
    }
}
