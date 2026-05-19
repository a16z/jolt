use crate::support::{FixtureId, FixtureMetadata};

pub fn metadata(id: FixtureId) -> FixtureMetadata {
    match id {
        FixtureId::MulDivSmall => FixtureMetadata {
            id,
            name: "muldiv small standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Small arithmetic/division standard fixture and default exhaustive tamper base.",
        },
        FixtureId::FibonacciSmall => FixtureMetadata {
            id,
            name: "fibonacci small standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Small loop/register fixture with short trace length.",
        },
        FixtureId::FibonacciMedium => FixtureMetadata {
            id,
            name: "fibonacci medium standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Same guest as fibonacci small with a longer trace.",
        },
        FixtureId::MemoryOps => FixtureMetadata {
            id,
            name: "memory ops standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "RAM fixture covering byte and halfword loads/stores.",
        },
        FixtureId::CollatzSmall => FixtureMetadata {
            id,
            name: "collatz small standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Branch-heavy variable-length loop fixture.",
        },
        FixtureId::Sha2Small => FixtureMetadata {
            id,
            name: "sha2 small standard",
            zk: false,
            has_trusted_advice: false,
            expected_core_accepts: true,
            notes: "Inline-heavy byte/hash fixture.",
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
        FixtureId::AdviceConsumer => FixtureMetadata {
            id,
            name: "advice consumer standard",
            zk: false,
            has_trusted_advice: true,
            expected_core_accepts: true,
            notes: "Core-backed guest that consumes both trusted and untrusted advice.",
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
