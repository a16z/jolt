//! Jolt-local Akita commitment configs.
//!
//! Configs delegate policy to Akita's proof-optimized presets while supplying
//! Jolt's generated schedule catalogs and setup sizing.

use akita_config::proof_optimized::fp128::{DenseBounded, OneHot};
use akita_config::{honest_fold_policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::GeneratedScheduleTable;
use akita_types::sis::CommittedSourceClass;
use akita_types::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey,
    FoldSchedule, OpeningClaimsLayout, SetupMatrixCapacity,
};

use crate::AKITA_ONE_HOT_K16;

fn dp_planned_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    let planned = akita_planner::find_schedule(
        key,
        honest_fold_policy_of::<Cfg>(),
        &[],
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

/// Fold one catalog row and its independently committed prefixes into `capacity`.
fn fold_row_capacity(
    capacity: &mut SetupMatrixCapacity,
    key: &AkitaScheduleLookupKey,
    schedule: impl FnOnce() -> Result<FoldSchedule, AkitaError>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError> {
    for precommitted in &key.precommitteds {
        if AkitaScheduleLookupKey::single(precommitted.group)
            .fits_setup_capacity(max_num_vars, max_num_batched_polys)?
        {
            let commit_only = commit_only_setup_field_elements(
                &precommitted.inner.matrix,
                &precommitted.outer.matrix,
                precommitted.outer_slice_count,
            )?;
            capacity.num_field_elements = capacity.num_field_elements.max(commit_only);
        }
    }

    if !key.fits_setup_capacity(max_num_vars, max_num_batched_polys)? {
        return Ok(());
    }
    let row_capacity = setup_matrix_capacity_for_schedule(&schedule()?)?;
    capacity.num_field_elements = capacity
        .num_field_elements
        .max(row_capacity.num_field_elements);
    Ok(())
}

/// Size setup for every cataloged or provisioned row within the advertised extent.
/// Schedule footprints are not monotone, so the maximum-shape row is insufficient.
fn catalog_setup_capacity<Cfg: CommitmentConfig + 'static>(
    table: &GeneratedScheduleTable,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixCapacity, AkitaError> {
    let fallback_key = AkitaScheduleLookupKey::single(
        OpeningClaimsLayout::new(max_num_vars, max_num_batched_polys)?.root_final_group_layout()?,
    );
    let mut capacity =
        setup_matrix_capacity_for_schedule(&dp_planned_schedule::<Cfg>(&fallback_key)?)?;
    for entry in table.entries {
        let key = entry.to_runtime_lookup_key();
        fold_row_capacity(
            &mut capacity,
            &key,
            // Catalog-only: this loop is already iterating the catalog, and
            // routing through the registry-aware hook would re-enter sizing.
            || Ok(crate::schedule_registry::catalog_only_row::<Cfg>(&key)?.into_schedule()),
            max_num_vars,
            max_num_batched_polys,
        )?;
    }
    for row in crate::schedule_registry::registered_rows::<Cfg>()?.rows() {
        let profiles = row.profiles();
        let key = AkitaScheduleLookupKey {
            final_group: profiles.final_group.group,
            precommitteds: profiles.precommitteds.clone(),
        };
        fold_row_capacity(
            &mut capacity,
            &key,
            || Ok(row.schedule().clone()),
            max_num_vars,
            max_num_batched_polys,
        )?;
    }
    Ok(capacity)
}

/// Delegates a [`CommitmentConfig`] to an upstream preset, overriding its
/// schedule catalog and catalog-backed setup sizing.
macro_rules! delegate_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $committed_source_class:expr,
        $catalog:expr
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

            fn decomposition() -> akita_types::DecompositionParams {
                <$base>::decomposition()
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_pcs::AkitaError>
            {
                <$base>::ring_challenge_config(d)
            }

            fn selection_policy() -> akita_schedules::SelectionPolicyId {
                <$base>::selection_policy()
            }

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                <$base>::sis_modulus_profile()
            }

            fn setup_matrix_capacity(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixCapacity, akita_pcs::AkitaError> {
                if max_num_batched_polys == 0 {
                    return Err(akita_pcs::AkitaError::InvalidSetup(
                        "max_num_batched_polys must be at least 1".to_string(),
                    ));
                }
                if let Some(table) = $catalog {
                    return catalog_setup_capacity::<Self>(
                        &table,
                        max_num_vars,
                        max_num_batched_polys,
                    );
                }
                let key = AkitaScheduleLookupKey::single(
                    akita_types::OpeningClaimsLayout::new(
                        max_num_vars,
                        max_num_batched_polys,
                    )?
                    .root_final_group_layout()?,
                );
                setup_matrix_capacity_for_schedule(&dp_planned_schedule::<Self>(&key)?)
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

            fn schedule_catalog() -> Option<akita_schedules::GeneratedScheduleTable> {
                $catalog
            }

            // Check provisioned advice rows before the static catalog. These overrides
            // mirror the trait defaults because an override cannot invoke its default body.
            fn resolve_catalog_row_for_key(
                key: &AkitaScheduleLookupKey,
            ) -> Result<akita_schedules::ResolvedScheduleRow, akita_pcs::AkitaError> {
                if let Some(row) = crate::schedule_registry::lookup_key::<Self>(key) {
                    return Ok(row);
                }
                Self::validate_sis_modulus_profile()?;
                akita_schedules::resolve_generated_catalog_row_for_key(
                    key,
                    &akita_config::policy_of::<Self>(),
                    Self::ring_challenge_config,
                    Self::schedule_catalog(),
                )
            }

            fn resolve_catalog_row_for_profiles(
                profiles: &akita_types::CommittedGroupBatchProfile,
            ) -> Result<akita_schedules::ResolvedScheduleRow, akita_pcs::AkitaError> {
                if let Some(row) = crate::schedule_registry::lookup_profiles::<Self>(profiles) {
                    return Ok(row);
                }
                Self::validate_sis_modulus_profile()?;
                profiles.validate(Self::decomposition().field_bits())?;
                akita_schedules::resolve_generated_catalog_row_for_profiles(
                    &AkitaScheduleLookupKey {
                        final_group: profiles.final_group.group,
                        precommitteds: profiles.precommitteds.clone(),
                    },
                    profiles,
                    &akita_config::policy_of::<Self>(),
                    Self::ring_challenge_config,
                    Self::schedule_catalog(),
                )
            }

            fn resolve_schedule_selection(
                selection: akita_types::OpeningScheduleSelection,
            ) -> Result<akita_schedules::ResolvedScheduleRow, akita_pcs::AkitaError> {
                if let Some(row) = crate::schedule_registry::lookup_selection::<Self>(selection) {
                    return Ok(row);
                }
                Self::validate_sis_modulus_profile()?;
                akita_schedules::resolve_generated_schedule_selection(
                    selection,
                    &akita_config::policy_of::<Self>(),
                    Self::ring_challenge_config,
                    Self::schedule_catalog(),
                )
            }
        }
    };
}

delegate_preset!(
    /// Adaptive one-hot config with the Jolt-generated K=16 schedule catalog.
    JoltOneHotK16,
    OneHot,
    CommittedSourceClass::UnitOneHot {
        source_chunk_size: AKITA_ONE_HOT_K16,
    },
    crate::schedules::jolt_fp128_onehot_k16_table()
);

delegate_preset!(
    /// Adaptive one-hot config with the Jolt-generated K=256 schedule catalog.
    JoltOneHotK256,
    OneHot,
    <OneHot as CommitmentConfig>::committed_source_class(),
    crate::schedules::jolt_fp128_onehot_k256_table()
);

delegate_preset!(
    /// Dense config for `u64`-bounded advice and committed-program objects.
    JoltDenseBounded,
    DenseBounded,
    <DenseBounded as CommitmentConfig>::committed_source_class(),
    crate::schedules::jolt_fp128_dense_bounded_table()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shapes_have_setup_capacities() {
        assert!(JoltDenseBounded::setup_matrix_capacity(14, 2).is_ok());
        assert!(JoltOneHotK16::setup_matrix_capacity(34, 1).is_ok());
        assert!(JoltOneHotK256::setup_matrix_capacity(43, 1).is_ok());
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn k256_policy_uses_adaptive_dimensions() {
        assert_eq!(JoltOneHotK256::inner_basis_range(), (3, 11));
        assert_eq!(JoltOneHotK256::opening_basis_range(), (3, 6));
        assert!(matches!(
            JoltOneHotK256::RING_DIMENSION_SCHEDULE_MODE,
            akita_schedules::RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));

        let layout = akita_types::OpeningClaimsLayout::new(39, 1).unwrap();
        let row = JoltOneHotK256::resolve_catalog_row_for_opening(&layout).unwrap();
        let schedule = row.schedule();
        let commitment = schedule.root.params.final_group();
        assert!([64, 128, 256, 512].contains(&commitment.profile.inner.matrix.ring_dimension()));
        assert!([64, 128].contains(&commitment.profile.outer.matrix.ring_dimension()));
    }
}
