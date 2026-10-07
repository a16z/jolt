//! Jolt-local Akita commitment configs.
//!
//! Configs contain protocol policy only. Schedule rows are supplied at runtime
//! as external `.aks` artifacts and are bound to an `AkitaCommitmentScheme`
//! instance.

use akita_config::proof_optimized::fp128::{Dense, DenseBounded, OneHot};
use akita_config::recursive_commitment::RecursiveScheduleConfig;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_prover::backend::SIGNED_BYTE_LOG_BASIS;
use akita_schedules::{RingDimensionScheduleMode, ADAPTIVE_SEARCH_LEVELS};
use akita_types::sis::CommittedSourceClass;
use akita_types::DecompositionParams;

use crate::AKITA_ONE_HOT_K16;

/// Delegate one Jolt policy to an upstream preset while assigning a distinct
/// external schedule-family identity.
macro_rules! delegate_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $committed_source_class:expr,
        $family_name:literal,
        $ring_dimension_schedule_mode:expr
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <$base as CommitmentConfig>::Field;
            type ExtField = <$base as CommitmentConfig>::ExtField;
            const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
                $ring_dimension_schedule_mode;
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
    /// Direct-planning policy used to generate the below-cutover K=16 rows.
    JoltOneHotK16Direct,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    "jolt-fp128-onehot-k16-direct-planner",
    <OneHot as CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE
);

delegate_preset!(
    /// Direct-planning policy used to generate the below-cutover K=256 rows.
    JoltOneHotK256Direct,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    "jolt-fp128-onehot-k256-direct-planner",
    // D128/rank 3 reduces the per-hot-entry work in both CPU and Metal commits.
    RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &[64],
        potential_a_dimensions: &[64, 128],
        potential_b_dimensions: &OneHot::B_RING_DIMENSIONS,
        potential_d_dimensions: &OneHot::D_RING_DIMENSIONS,
    }
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

delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
    "jolt-fp128-dense-bounded",
    <DenseBounded as CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE
);

/// A balanced signed base-2^8 source committed as its own digit planes, so
/// the root reads the stored bytes: the A basis is pinned to one byte, and
/// the A domain keeps D64 because the adaptive policy requires its suffix
/// dimension there.
macro_rules! byte_digit_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $family_name:literal,
        $log_commit_bound:expr
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <Dense as CommitmentConfig>::Field;
            type ExtField = <Dense as CommitmentConfig>::ExtField;
            const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
                RingDimensionScheduleMode::AdaptiveDimension {
                    num_search_levels: ADAPTIVE_SEARCH_LEVELS,
                    suffix_dimensions: &[64],
                    potential_a_dimensions: &[64, 128],
                    potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
                    potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
                };

            fn schedule_family_name() -> &'static str {
                $family_name
            }

            fn decomposition() -> DecompositionParams {
                DecompositionParams {
                    log_basis: Dense::decomposition().log_basis,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: Some(128),
                }
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_pcs::AkitaError> {
                Dense::ring_challenge_config(d)
            }

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                Dense::sis_modulus_profile()
            }

            fn opening_basis_range() -> (u32, u32) {
                Dense::opening_basis_range()
            }

            fn inner_basis_range() -> (u32, u32) {
                (SIGNED_BYTE_LOG_BASIS, SIGNED_BYTE_LOG_BASIS)
            }

            fn committed_source_class() -> CommittedSourceClass {
                CommittedSourceClass::BalancedSignedDigit
            }
        }
    };
}

byte_digit_preset!(
    /// Direct-planning policy of the signed-byte trace `Q`: one digit per
    /// coefficient.
    JoltSignedBytesDirect,
    "jolt-fp128-signed-bytes-direct-planner",
    SIGNED_BYTE_LOG_BASIS
);

byte_digit_preset!(
    /// Direct-planning policy of full field values committed as sixteen
    /// signed-byte digit planes.
    JoltFieldDigitsDirect,
    "jolt-fp128-field-digits-direct-planner",
    128
);

impl RecursiveScheduleConfig for JoltSignedBytesDirect {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-signed-bytes";
}

impl RecursiveScheduleConfig for JoltFieldDigitsDirect {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "jolt-fp128-field-digits";
}

/// Runtime signed-byte trace policy; like the one-hot families, its catalog
/// holds direct rows below the trace cutover and setup-offloaded rows above.
pub type JoltSignedBytes = RecursiveCommitmentConfig<JoltSignedBytesDirect>;

/// Runtime policy of the field-digit groups opened beside `Q`.
pub type JoltFieldDigits = RecursiveCommitmentConfig<JoltFieldDigitsDirect>;

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
            RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));
        assert!(JoltOneHotK16::recursive_setup_planning());
        assert!(JoltOneHotK256::recursive_setup_planning());
    }
}
