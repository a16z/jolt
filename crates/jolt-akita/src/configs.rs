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
    setup_matrix_envelope_for_schedule, AkitaScheduleLookupKey, SetupMatrixEnvelope,
};

/// Sizes a production OneHotTrace setup directly from the checked-in Jolt catalog.
///
/// `Some` means the requested maximum shape itself is catalog-backed. Smaller
/// catalog rows are included because setup matrices are shared prefix views
/// and planned footprints are not monotone in either layout dimension.
fn catalog_setup_envelope<Cfg: CommitmentConfig>(
    table: GeneratedScheduleTable,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Option<SetupMatrixEnvelope>, AkitaError> {
    let requested_shape_is_catalogued = table.entries.iter().any(|entry| {
        entry.root.precommitted_groups.is_empty()
            && entry.root.final_group.layout.num_vars() == max_num_vars
            && entry.root.final_group.layout.num_polynomials() == max_num_batched_polys
    });
    if !requested_shape_is_catalogued {
        return Ok(None);
    }

    let mut envelope = SetupMatrixEnvelope::minimum();
    for entry in table.entries.iter().filter(|entry| {
        entry.root.precommitted_groups.is_empty()
            && entry.root.final_group.layout.num_vars() <= max_num_vars
            && entry.root.final_group.layout.num_polynomials() <= max_num_batched_polys
    }) {
        let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            entry.root.final_group.layout,
        ))?;
        let entry_envelope = setup_matrix_envelope_for_schedule(&schedule)?;
        envelope.max_setup_len = envelope.max_setup_len.max(entry_envelope.max_setup_len);
    }
    Ok(Some(envelope))
}

/// Delegates a [`CommitmentConfig`] to an upstream preset, overriding its
/// schedule catalog and catalog-backed setup sizing. `get_params_for_prove`
/// re-derives the single-group lookup key through the public layout API;
/// multi-group layouts (never produced by Jolt's shapes) fall back to the base
/// preset's DP planning.
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

            fn ring_subfield_embedding_norm_bound() -> u32 {
                <$base>::ring_subfield_embedding_norm_bound()
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_pcs::AkitaError> {
                if max_num_batched_polys == 0 {
                    return Err(akita_pcs::AkitaError::InvalidSetup(
                        "max_num_batched_polys must be at least 1".to_string(),
                    ));
                }
                if let Some(table) = $catalog {
                    if let Some(envelope) = catalog_setup_envelope::<Self>(
                        table,
                        max_num_vars,
                        max_num_batched_polys,
                    )? {
                        return Ok(envelope);
                    }
                }
                <$base>::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
            }

            fn basis_range() -> (u32, u32) {
                <$base>::basis_range()
            }

            fn onehot_chunk_size() -> usize {
                <$base>::onehot_chunk_size()
            }

            fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
                <$base>::chunked_witness_cfg()
            }

            fn recursive_setup_planning() -> bool {
                <$base>::recursive_setup_planning()
            }

            fn supports_multi_group_final_commit() -> bool {
                <$base>::supports_multi_group_final_commit()
            }

            fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
                $catalog
            }

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::FoldSchedule, akita_pcs::AkitaError> {
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "catalog setup tests should fail loudly on malformed schedules"
)]
mod tests {
    use super::*;

    #[test]
    fn production_one_hot_trace_shapes_use_catalog_setup_sizing() {
        let k16 = crate::schedules::jolt_fp128_d64_onehot_k16_table().unwrap();
        assert!(catalog_setup_envelope::<JoltD64OneHotK16>(k16, 28, 81)
            .unwrap()
            .is_some());

        let k256 = crate::schedules::jolt_fp128_d64_onehot_k256_table().unwrap();
        assert!(catalog_setup_envelope::<JoltD64OneHotK256>(k256, 38, 41)
            .unwrap()
            .is_some());
    }

    #[test]
    fn setup_sizing_rejects_zero_batched_polynomials() {
        let err = JoltD64OneHotK16::max_setup_matrix_size(4, 0)
            .expect_err("zero batched polynomials must be rejected");
        assert!(
            err.to_string().contains("at least 1"),
            "unexpected error: {err}"
        );
    }

    /// A shape outside the checked-in catalog grid must fall back to the
    /// upstream preset's sizing rather than inventing an envelope:
    /// poly-count 2 is not a reachable `OneHotTrace` width in either family,
    /// so the delegated outcome (envelope or sizing error) must equal the
    /// base preset's, dimension by dimension.
    #[test]
    #[expect(
        clippy::panic,
        reason = "delegation must not turn a base sizing error into an envelope or vice versa"
    )]
    fn non_catalogued_shape_falls_back_to_upstream_preset_sizing() {
        for max_num_vars in [6, 17] {
            let k16 = crate::schedules::jolt_fp128_d64_onehot_k16_table().unwrap();
            assert!(
                catalog_setup_envelope::<JoltD64OneHotK16>(k16, max_num_vars, 2)
                    .unwrap()
                    .is_none(),
                "poly-count 2 must not be treated as catalogued"
            );

            let delegated = JoltD64OneHotK16::max_setup_matrix_size(max_num_vars, 2);
            let base = akita_config::proof_optimized::fp128::D64OneHotK16::max_setup_matrix_size(
                max_num_vars,
                2,
            );
            match (delegated, base) {
                (Ok(delegated), Ok(base)) => assert_eq!(
                    delegated.max_setup_len, base.max_setup_len,
                    "fallback must delegate to the base preset's envelope"
                ),
                (Err(delegated), Err(base)) => assert_eq!(
                    delegated.to_string(),
                    base.to_string(),
                    "fallback must surface the base preset's sizing error"
                ),
                (delegated, base) => panic!(
                    "fallback diverged from the base preset at {max_num_vars} vars: \
                     delegated={delegated:?} base={base:?}"
                ),
            }
        }
    }
}
