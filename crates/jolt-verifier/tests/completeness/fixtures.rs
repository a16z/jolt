use crate::support::{FixtureId, FixtureMetadata};

pub fn metadata(id: FixtureId) -> FixtureMetadata {
    match id {
        FixtureId::MulDivSmall => FixtureMetadata {
            id,
            name: "muldiv small standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Live-generated small muldiv proof once fixture generation is wired.",
        },
        FixtureId::MulDivZkSmall => FixtureMetadata {
            id,
            name: "muldiv small ZK",
            zk: true,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Live-generated small ZK muldiv proof once fixture generation is wired.",
        },
        FixtureId::ZkStage1Prefix => FixtureMetadata {
            id,
            name: "ZK Stage 1 prefix",
            zk: true,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes:
                "Prefix BlindFold fixture for the Stage 1 verifier frontier; not a full core proof.",
        },
        FixtureId::AdviceCommitments => FixtureMetadata {
            id,
            name: "advice commitments standard",
            zk: false,
            has_trusted_advice: true,
            expected_core_accepts: true,
            notes: "Core-backed proof with trusted and untrusted advice commitments.",
        },
        FixtureId::PublicIoMismatch
        | FixtureId::TrustedAdviceMismatch
        | FixtureId::MixedProofMode
        | FixtureId::ConfigMismatch
        | FixtureId::OpeningClaimMismatch
        | FixtureId::BlindFoldMismatch => FixtureMetadata {
            id,
            name: "soundness-only fixture",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: false,
            notes: "Reserved for soundness tests.",
        },
    }
}
