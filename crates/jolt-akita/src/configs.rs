//! Jolt-local Akita commitment configs.
//!
//! Configs contain protocol policy only. Schedule rows are supplied at runtime
//! as external `.aks` artifacts and are bound to an `AkitaCommitmentScheme`
//! instance.

use akita_config::proof_optimized::fp128::{DenseBounded, OneHot};
use akita_config::CommitmentConfig;
use akita_types::sis::CommittedSourceClass;

use crate::AKITA_ONE_HOT_K16;

/// Delegate one Jolt policy to an upstream preset while assigning a distinct
/// external schedule-family identity.
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
                <$base>::chunked_witness_cfg()
            }

            fn recursive_setup_planning() -> bool {
                <$base>::recursive_setup_planning()
            }
        }
    };
}

delegate_preset!(
    /// Adaptive one-hot config using the Jolt K=16 schedule artifact.
    JoltOneHotK16,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    "jolt-fp128-onehot-k16"
);

delegate_preset!(
    /// Adaptive one-hot config using the Jolt K=256 schedule artifact.
    JoltOneHotK256,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    "jolt-fp128-onehot-k256"
);

delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
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
    }
}
