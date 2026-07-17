//! Jolt-local Akita commitment configs.
//!
//! Each config delegates every policy decision (field, ring, decomposition,
//! SIS profile, chunking) to its upstream proof-optimized preset and
//! overrides exactly one hook: [`akita_config::CommitmentConfig::schedule_catalog`],
//! so the generated schedule tables for Jolt's `W_jolt` shapes live in this
//! crate (see [`crate::schedules`]) while the planner policy keeps one
//! upstream owner. The catalog is identity-validated against the config's
//! policy on every lookup, so a policy/table drift hard-errors instead of
//! silently planning a different schedule.

use akita_config::CommitmentConfig;

/// Delegates a [`CommitmentConfig`] to an upstream preset, overriding only
/// the schedule catalog. `get_params_for_prove` re-derives the single-group
/// lookup key through the public layout API; multi-group layouts (never
/// produced by Jolt's shapes) fall back to the base preset's DP planning.
macro_rules! delegate_preset {
    ($(#[$doc:meta])* $name:ident, $base:ty, $catalog:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <$base as CommitmentConfig>::Field;
            type ExtField = <$base as CommitmentConfig>::ExtField;
            const D: usize = <$base as CommitmentConfig>::D;

            fn decomposition() -> akita_types::DecompositionParams {
                <$base>::decomposition()
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_pcs::AkitaError>
            {
                <$base>::ring_challenge_config(d)
            }

            fn fold_challenge_shape_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> akita_challenges::TensorChallengeShape {
                <$base>::fold_challenge_shape_at_level(inputs)
            }

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base>::sis_modulus_profile()
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_pcs::AkitaError> {
                <$base>::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
            }

            fn basis_range() -> (u32, u32) {
                <$base>::basis_range()
            }

            fn onehot_chunk_size() -> usize {
                <$base>::onehot_chunk_size()
            }

            fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
                $catalog
            }

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::Schedule, akita_pcs::AkitaError> {
                if layout.num_groups() == 1 {
                    layout.check()?;
                    Self::runtime_schedule(akita_types::AkitaScheduleLookupKey::single(
                        layout.root_final_group_layout()?,
                    ))
                } else {
                    <$base>::get_params_for_prove(layout)
                }
            }
        }
    };
}

delegate_preset!(
    /// `D64OneHotK16` with the Jolt-generated K=16 schedule catalog.
    JoltD64OneHotK16,
    akita_config::proof_optimized::fp128::D64OneHotK16,
    crate::schedules::jolt_fp128_d64_onehot_k16_table()
);

delegate_preset!(
    /// `D64OneHot` (K=256) with the Jolt-generated large-trace catalog.
    JoltD64OneHotK256,
    akita_config::proof_optimized::fp128::D64OneHot,
    crate::schedules::jolt_fp128_d64_onehot_k256_table()
);
