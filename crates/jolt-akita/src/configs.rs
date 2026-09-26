//! Jolt-local Akita commitment configs.
//!
//! Configs contain protocol policy only. Schedule rows are supplied at runtime
//! as external `.aks` artifacts and are bound to an `AkitaCommitmentScheme`
//! instance.

use akita_config::proof_optimized::fp128::{DenseBounded, OneHot};
use akita_config::recursive_commitment::RecursiveScheduleConfig;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_types::sis::CommittedSourceClass;
use akita_types::{ChunkedWitnessCfg, MultiChunkProfileId};

use crate::AKITA_ONE_HOT_K16;

/// Delegate one Jolt policy to an upstream preset while assigning a distinct
/// external schedule-family identity.
macro_rules! delegate_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $committed_source_class:expr,
        $chunked_witness_cfg:expr,
        $family_name:literal
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <$base as CommitmentConfig>::Field;
            type ExtField = <$base as CommitmentConfig>::ExtField;
            const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
                <$base as CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE;
            const EXT_DEGREE: usize = <$base as CommitmentConfig>::EXT_DEGREE;

            fn schedule_family_name() -> &'static str {
                $family_name
            }

            fn decomposition() -> akita_types::DecompositionParams {
                <$base>::decomposition()
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_pcs::AkitaError> {
                <$base>::ring_challenge_config(d)
            }

            fn selection_policy() -> akita_schedules::SelectionPolicyId {
                <$base>::selection_policy()
            }

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base>::sis_modulus_profile()
            }

            fn opening_basis_range() -> (u32, u32) {
                <$base>::opening_basis_range()
            }

            fn inner_basis_range() -> (u32, u32) {
                <$base>::inner_basis_range()
            }

            fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
                $committed_source_class
            }

            fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
                $chunked_witness_cfg
            }

            fn recursive_setup_planning() -> bool {
                <$base>::recursive_setup_planning()
            }
        }
    };
}

delegate_preset!(
    /// Direct-planning policy used to generate the below-cutover K=16 rows.
    JoltOneHotK16Direct,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    ChunkedWitnessCfg::default_non_chunked(),
    "jolt-fp128-onehot-k16-direct-planner"
);

delegate_preset!(
    /// Direct-planning policy used to generate the below-cutover K=256 rows.
    JoltOneHotK256Direct,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    ChunkedWitnessCfg::default_non_chunked(),
    "jolt-fp128-onehot-k256-direct-planner"
);

impl RecursiveScheduleConfig for JoltOneHotK16Direct {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k16";
}

impl RecursiveScheduleConfig for JoltOneHotK256Direct {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k256";
}

/// Runtime K=16 policy. Its catalog may contain either direct or setup-offloaded
/// rows; the exact admitted row decides the contribution mode for each shape.
pub type JoltOneHotK16 = RecursiveCommitmentConfig<JoltOneHotK16Direct>;

/// Runtime K=256 policy. Its catalog may contain either direct or setup-offloaded
/// rows; the exact admitted row decides the contribution mode for each shape.
pub type JoltOneHotK256 = RecursiveCommitmentConfig<JoltOneHotK256Direct>;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum AkitaOneHotChunkProfile {
    #[default]
    Single = 0,
    Two = 1,
    Four = 2,
    Eight = 3,
}

impl AkitaOneHotChunkProfile {
    pub const fn num_chunks(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

macro_rules! chunked_one_hot_config {
    (
        $direct:ident,
        $recursive:ident,
        $base:ty,
        $committed_source_class:expr,
        $profile:expr,
        $direct_family:literal,
        $recursive_family:literal
    ) => {
        delegate_preset!(
            $direct,
            $base,
            $committed_source_class,
            ChunkedWitnessCfg::from_profile($profile),
            $direct_family
        );

        impl RecursiveScheduleConfig for $direct {
            const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = $recursive_family;
        }

        pub type $recursive = RecursiveCommitmentConfig<$direct>;
    };
}

chunked_one_hot_config!(
    JoltOneHotK16W2R2Direct,
    JoltOneHotK16W2R2,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    MultiChunkProfileId::W2R2,
    "jolt-fp128-onehot-k16-w2r2-direct-planner",
    "jolt-fp128-onehot-k16-w2r2"
);

chunked_one_hot_config!(
    JoltOneHotK256W2R2Direct,
    JoltOneHotK256W2R2,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    MultiChunkProfileId::W2R2,
    "jolt-fp128-onehot-k256-w2r2-direct-planner",
    "jolt-fp128-onehot-k256-w2r2"
);

chunked_one_hot_config!(
    JoltOneHotK16W4R2Direct,
    JoltOneHotK16W4R2,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    MultiChunkProfileId::W4R2,
    "jolt-fp128-onehot-k16-w4r2-direct-planner",
    "jolt-fp128-onehot-k16-w4r2"
);

chunked_one_hot_config!(
    JoltOneHotK256W4R2Direct,
    JoltOneHotK256W4R2,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    MultiChunkProfileId::W4R2,
    "jolt-fp128-onehot-k256-w4r2-direct-planner",
    "jolt-fp128-onehot-k256-w4r2"
);

delegate_preset!(
    /// Multi-chunk companion for K=16 trace openings.
    JoltOneHotK16MultiChunkDirect,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    ChunkedWitnessCfg::d64_production(),
    "jolt-fp128-onehot-k16-multi-chunk-direct-planner"
);

delegate_preset!(
    /// Multi-chunk companion for K=256 trace openings.
    JoltOneHotK256MultiChunkDirect,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    ChunkedWitnessCfg::d64_production(),
    "jolt-fp128-onehot-k256-multi-chunk-direct-planner"
);

impl RecursiveScheduleConfig for JoltOneHotK16MultiChunkDirect {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k16-multi-chunk";
}

impl RecursiveScheduleConfig for JoltOneHotK256MultiChunkDirect {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k256-multi-chunk";
}

pub type JoltOneHotK16MultiChunk = RecursiveCommitmentConfig<JoltOneHotK16MultiChunkDirect>;
pub type JoltOneHotK256MultiChunk = RecursiveCommitmentConfig<JoltOneHotK256MultiChunkDirect>;

delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
    ChunkedWitnessCfg::default_non_chunked(),
    "jolt-fp128-dense-bounded"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jolt_families_are_distinct() {
        assert_ne!(
            JoltDenseBounded::schedule_family_name(),
            JoltOneHotK16::schedule_family_name()
        );
        assert_ne!(
            JoltOneHotK16::schedule_family_name(),
            JoltOneHotK256::schedule_family_name()
        );
    }

    #[test]
    fn k256_policy_uses_adaptive_dimensions() {
        assert_eq!(JoltOneHotK256::inner_basis_range(), (3, 16));
        assert_eq!(JoltOneHotK256::opening_basis_range(), (3, 6));
        assert!(matches!(
            JoltOneHotK256::RING_DIMENSION_SCHEDULE_MODE,
            akita_schedules::RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));
        assert!(JoltOneHotK16::recursive_setup_planning());
        assert!(JoltOneHotK256::recursive_setup_planning());
        assert_eq!(
            JoltOneHotK16MultiChunk::chunked_witness_cfg(),
            ChunkedWitnessCfg::d64_production()
        );
        assert_eq!(
            JoltOneHotK256MultiChunk::chunked_witness_cfg(),
            ChunkedWitnessCfg::d64_production()
        );
        for actual in [
            JoltOneHotK16W2R2::chunked_witness_cfg(),
            JoltOneHotK256W2R2::chunked_witness_cfg(),
        ] {
            assert_eq!(
                actual,
                ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W2R2)
            );
        }
        for actual in [
            JoltOneHotK16W4R2::chunked_witness_cfg(),
            JoltOneHotK256W4R2::chunked_witness_cfg(),
        ] {
            assert_eq!(
                actual,
                ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W4R2)
            );
        }
        assert_eq!(
            JoltOneHotK16::chunked_witness_cfg(),
            ChunkedWitnessCfg::default_non_chunked()
        );
        assert_eq!(
            JoltOneHotK256::chunked_witness_cfg(),
            ChunkedWitnessCfg::default_non_chunked()
        );
        assert_eq!(
            JoltDenseBounded::chunked_witness_cfg(),
            ChunkedWitnessCfg::default_non_chunked()
        );
    }
}
