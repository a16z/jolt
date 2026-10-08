//! Jolt-local Akita commitment configs.
//!
//! Configs contain protocol policy only. Schedule rows are supplied at runtime
//! as external `.aks` artifacts and are bound to an `AkitaCommitmentScheme`
//! instance.

use akita_config::proof_optimized::fp128::{Dense, DenseBounded, OneHot};
use akita_config::recursive_commitment::RecursiveScheduleConfig;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_params::sis::CommittedSourceClass;
use akita_params::{ChunkedWitnessCfg, MultiChunkProfileId};

use crate::AKITA_ONE_HOT_K16;

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

            fn decomposition() -> akita_params::DecompositionParams {
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

            fn sis_modulus_profile() -> akita_params::SisModulusProfileId {
                <$base>::sis_modulus_profile()
            }

            fn opening_basis_range() -> (u32, u32) {
                <$base>::opening_basis_range()
            }

            fn inner_basis_range() -> (u32, u32) {
                <$base>::inner_basis_range()
            }

            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                $committed_source_class
            }

            fn chunked_witness_cfg() -> akita_params::ChunkedWitnessCfg {
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
pub enum AkitaChunkProfile {
    #[default]
    Single = 0,
    Two = 1,
    Four = 2,
    Eight = 3,
}

impl AkitaChunkProfile {
    pub(crate) const fn witness_cfg(self) -> ChunkedWitnessCfg {
        match self {
            Self::Single => ChunkedWitnessCfg::default_non_chunked(),
            Self::Two => ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W2R2),
            Self::Four => ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W4R2),
            Self::Eight => ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W8R2),
        }
    }

    pub const fn num_chunks(self) -> usize {
        self.witness_cfg().num_chunks
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
            $profile.witness_cfg(),
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
    AkitaChunkProfile::Two,
    "jolt-fp128-onehot-k16-w2r2-direct-planner",
    "jolt-fp128-onehot-k16-w2r2"
);

chunked_one_hot_config!(
    JoltOneHotK256W2R2Direct,
    JoltOneHotK256W2R2,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    AkitaChunkProfile::Two,
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
    AkitaChunkProfile::Four,
    "jolt-fp128-onehot-k16-w4r2-direct-planner",
    "jolt-fp128-onehot-k16-w4r2"
);

chunked_one_hot_config!(
    JoltOneHotK256W4R2Direct,
    JoltOneHotK256W4R2,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    AkitaChunkProfile::Four,
    "jolt-fp128-onehot-k256-w4r2-direct-planner",
    "jolt-fp128-onehot-k256-w4r2"
);

delegate_preset!(
    /// W8R2 companion for K=16 trace openings.
    JoltOneHotK16W8R2Direct,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    AkitaChunkProfile::Eight.witness_cfg(),
    "jolt-fp128-onehot-k16-w8r2-direct-planner"
);

delegate_preset!(
    /// W8R2 companion for K=256 trace openings.
    JoltOneHotK256W8R2Direct,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    AkitaChunkProfile::Eight.witness_cfg(),
    "jolt-fp128-onehot-k256-w8r2-direct-planner"
);

impl RecursiveScheduleConfig for JoltOneHotK16W8R2Direct {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k16-w8r2";
}

impl RecursiveScheduleConfig for JoltOneHotK256W8R2Direct {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-onehot-k256-w8r2";
}

pub type JoltOneHotK16W8R2 = RecursiveCommitmentConfig<JoltOneHotK16W8R2Direct>;
pub type JoltOneHotK256W8R2 = RecursiveCommitmentConfig<JoltOneHotK256W8R2Direct>;

// Dense honest sizing retains the unchunked response cap in every chunk, so
// the largest supported count also gives the largest A collision envelope.
// Freeze this producer policy independently of the consuming trace profile.
delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    /// Certifies producers for the maximum supported response chunk count.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
    AkitaChunkProfile::Eight.witness_cfg(),
    "jolt-fp128-dense-bounded"
);

delegate_preset!(
    /// Dense config for arbitrary field values, including field-register increments.
    /// Certifies producers for the maximum supported response chunk count.
    JoltDenseFull,
    Dense,
    <Dense as CommitmentConfig>::committed_source_class(),
    AkitaChunkProfile::Eight.witness_cfg(),
    "jolt-fp128-dense-full"
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
}
