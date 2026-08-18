//! Jolt-local Akita commitment configs.
//!
//! Each config delegates every policy decision (field, ring, decomposition,
//! SIS profile, chunking) to its upstream proof-optimized preset and
//! overrides the schedule catalog and setup sizing hooks,
//! so the generated schedule tables for Jolt's `OneHotTrace` shapes live in this
//! crate (see [`crate::schedules`]) while the planner policy keeps one
//! upstream owner. The catalog is identity-validated against the config's
//! policy on every lookup, so a policy/table drift hard-errors instead of
//! silently planning a different schedule.

use akita_config::CommitmentConfig;
use akita_pcs::AkitaError;
use akita_planner::GeneratedScheduleTable;
use akita_types::{
    setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey, SetupMatrixCapacity,
};

fn dp_planned_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<akita_types::FoldSchedule, AkitaError> {
    let planned = akita_planner::find_schedule(
        key,
        Cfg::root_honest_fold_policy(),
        &[],
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

/// Sizes a production OneHotTrace setup directly from the checked-in Jolt catalog.
///
/// `Some` means the requested maximum shape itself is catalog-backed. Smaller
/// catalog rows are included because setup matrices are shared prefix views
/// and planned footprints are not monotone in either layout dimension.
fn catalog_setup_capacity<Cfg: CommitmentConfig>(
    table: &GeneratedScheduleTable,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Option<SetupMatrixCapacity>, AkitaError> {
    let requested_shape_is_catalogued = table.entries.iter().any(|entry| {
        entry.root.precommitted_groups.is_empty()
            && entry.root.final_group.layout.num_vars() == max_num_vars
            && entry.root.final_group.layout.num_polynomials() == max_num_batched_polys
    });
    if !requested_shape_is_catalogued {
        return Ok(None);
    }

    let mut capacity = SetupMatrixCapacity::minimum();
    for entry in table.entries.iter().filter(|entry| {
        entry.root.precommitted_groups.is_empty()
            && entry.root.final_group.layout.num_vars() <= max_num_vars
            && entry.root.final_group.layout.num_polynomials() <= max_num_batched_polys
    }) {
        let row = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            entry.root.final_group.layout,
        ))?;
        let entry_capacity = setup_matrix_capacity_for_schedule(row.schedule())?;
        capacity.num_field_elements = capacity
            .num_field_elements
            .max(entry_capacity.num_field_elements);
    }
    Ok(Some(capacity))
}

/// Delegates a [`CommitmentConfig`] to an upstream preset, overriding its
/// schedule catalog and catalog-backed setup sizing.
macro_rules! delegate_preset {
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $root_honest_fold_policy:expr,
        $catalog:expr
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl CommitmentConfig for $name {
            type Field = <$base as CommitmentConfig>::Field;
            type ExtField = <$base as CommitmentConfig>::ExtField;
            const D: usize = <$base as CommitmentConfig>::D;
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
                    if let Some(capacity) = catalog_setup_capacity::<Self>(
                        &table,
                        max_num_vars,
                        max_num_batched_polys,
                    )? {
                        return Ok(capacity);
                    }
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

            fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
                $root_honest_fold_policy
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
        }
    };
}

delegate_preset!(
    /// Adaptive one-hot config with the Jolt-generated K=16 schedule catalog.
    JoltOneHotK16,
    akita_config::proof_optimized::fp128::OneHot,
    akita_types::sis::HonestFoldPolicySpec::UnitOneHot(
        akita_types::sis::UnitOneHotFoldPolicy::new(128, 1, 16),
    ),
    crate::schedules::jolt_fp128_onehot_k16_table()
);

delegate_preset!(
    /// Adaptive one-hot config with the Jolt-generated K=256 schedule catalog.
    JoltOneHotK256,
    akita_config::proof_optimized::fp128::OneHot,
    akita_config::proof_optimized::fp128::OneHot::root_honest_fold_policy(),
    crate::schedules::jolt_fp128_onehot_k256_table()
);

delegate_preset!(
    /// Adaptive dense config with the Jolt-generated advice/program byte-object catalog.
    JoltDense,
    akita_config::proof_optimized::fp128::Dense,
    akita_config::proof_optimized::fp128::Dense::root_honest_fold_policy(),
    crate::schedules::jolt_fp128_dense_table()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shapes_have_setup_capacities() {
        assert!(JoltDense::setup_matrix_capacity(14, 2).is_ok());
        assert!(JoltOneHotK16::setup_matrix_capacity(34, 1).is_ok());
        assert!(JoltOneHotK256::setup_matrix_capacity(43, 1).is_ok());
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn k256_policy_uses_adaptive_dimensions() {
        assert_eq!(JoltOneHotK256::D, 256);
        assert_eq!(JoltOneHotK256::inner_basis_range(), (3, 11));
        assert_eq!(JoltOneHotK256::opening_basis_range(), (3, 6));
        assert!(matches!(
            JoltOneHotK256::RING_DIMENSION_SCHEDULE_MODE,
            akita_schedules::RingDimensionScheduleMode::AdaptiveDimension { .. }
        ));

        let layout = akita_types::OpeningClaimsLayout::new(39, 1).unwrap();
        let row = JoltOneHotK256::resolve_catalog_row_for_opening(&layout).unwrap();
        let schedule = row.schedule();
        let commitment = &schedule.root.params.final_group.commitment;
        assert!([64, 128, 256].contains(&commitment.inner_commit_matrix.ring_dimension()));
        assert!([64, 128].contains(&commitment.outer_commit_matrix.ring_dimension()));
    }
}
