use crate::support::{FixtureId, FixtureMetadata};

pub fn metadata(id: FixtureId) -> FixtureMetadata {
    match id {
        FixtureId::ClearBase => FixtureMetadata {
            id,
            name: "clear base Dory-assist fixture",
            zk: false,
            expected_accepts: true,
            notes: "Canonical clear-mode fixture used as the prover/verifier completeness oracle.",
        },
        FixtureId::ClearMultiround => FixtureMetadata {
            id,
            name: "clear multi-round Dory-assist fixture",
            zk: false,
            expected_accepts: true,
            notes: "Clear-mode fixture with a two-round Dory-reduce proof used to pin multi-round verifier completeness.",
        },
        FixtureId::ZkBase => FixtureMetadata {
            id,
            name: "ZK base Dory-assist fixture",
            zk: true,
            expected_accepts: true,
            notes: "Canonical ZK-mode fixture used as the prover/verifier completeness oracle.",
        },
        FixtureId::ZkMultiround => FixtureMetadata {
            id,
            name: "ZK multi-round Dory-assist fixture",
            zk: true,
            expected_accepts: true,
            notes: "ZK-mode fixture with a two-round Dory-reduce proof used to pin multi-round verifier completeness.",
        },
        _ => FixtureMetadata {
            id,
            name: "soundness-only fixture",
            zk: matches!(id, FixtureId::ZkInputMismatch),
            expected_accepts: false,
            notes: "Reserved for Dory-assist soundness tests.",
        },
    }
}
