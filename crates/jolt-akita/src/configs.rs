//! Jolt-local Akita commitment configs.
//!
//! Configs contain protocol policy only. Schedule rows are supplied at runtime
//! as external `.aks` artifacts and are bound to an `AkitaCommitmentScheme`
//! instance.

use akita_config::proof_optimized::fp128::{Dense, DenseBounded, OneHot};
use akita_config::recursive_commitment::RecursiveScheduleConfig;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_params::sis::CommittedSourceClass;
use akita_params::ChunkedWitnessCfg;

use std::fmt::{Display, Formatter, Result as FormatResult};

use crate::AKITA_ONE_HOT_K16;

macro_rules! delegate_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $committed_source_class:expr,
        $family_name:literal
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <$base as CommitmentConfig>::Field;
            type ExtField = <$base as CommitmentConfig>::ExtField;
            const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
                <$base as CommitmentConfig>::RING_DIMENSION_SCHEDULE;
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
    "jolt-fp128-onehot-k16-direct-planner"
);

delegate_preset!(
    /// Direct-planning policy used to generate the below-cutover K=256 rows.
    JoltOneHotK256Direct,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
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

/// Witness geometry selected by trusted preprocessing configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaChunkProfile {
    pub chunks: usize,
    pub levels: usize,
}

impl Display for AkitaChunkProfile {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        write!(f, "w{}r{}", self.chunks, self.levels)
    }
}

impl Default for AkitaChunkProfile {
    fn default() -> Self {
        Self::new(1, 0)
    }
}

impl AkitaChunkProfile {
    pub const fn new(chunks: usize, levels: usize) -> Self {
        Self { chunks, levels }
    }

    pub const fn witness_cfg(self) -> ChunkedWitnessCfg {
        ChunkedWitnessCfg {
            num_chunks: self.chunks,
            num_activated_levels: self.levels,
        }
    }

    pub const fn num_chunks(self) -> usize {
        self.chunks
    }
}

delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
    "jolt-fp128-dense-bounded"
);

delegate_preset!(
    /// Dense config for arbitrary field values, including field-register increments.
    JoltDenseFull,
    Dense,
    <Dense as CommitmentConfig>::committed_source_class(),
    "jolt-fp128-dense-full"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_display_is_a_canonical_csv_field() {
        for (profile, expected) in [
            (AkitaChunkProfile::default(), "w1r0"),
            (AkitaChunkProfile::new(2, 3), "w2r3"),
            (AkitaChunkProfile::new(8, 4), "w8r4"),
        ] {
            assert_eq!(profile.to_string(), expected);
        }
    }

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
