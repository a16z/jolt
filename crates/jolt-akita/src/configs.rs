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

/// Planner-DP fallback shared by runtime schedule resolution and setup
/// sizing: both MUST plan under the same policy or envelopes drift from the
/// schedules that later run against them.
fn dp_planned_schedule<Cfg: akita_config::CommitmentConfig>(
    key: &akita_types::AkitaScheduleLookupKey,
) -> Result<akita_types::FoldSchedule, akita_pcs::AkitaError> {
    let planned = akita_planner::find_group_batch_schedule(
        key,
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
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
    ($(#[$doc:meta])* $name:ident, $base:ty, $catalog:expr, $slack_permille:expr) => {
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
                // Catalog miss: size from a DP-planned schedule for the
                // exact requested shape, mirroring `runtime_schedule`'s
                // fallback under THIS config's policy. Never consult the base
                // preset's catalog here — its rows are planned under the base
                // policy and can disagree with the schedule this config's
                // runtime resolution will actually produce. Jolt setups are
                // exactly sized (`validate_commit_shape` pins num_vars to the
                // setup dimension), so the single-shape envelope is the one
                // the runtime fallback schedule will need.
                let key = akita_types::AkitaScheduleLookupKey::single(
                    akita_types::OpeningClaimsLayout::new(
                        max_num_vars,
                        max_num_batched_polys,
                    )?
                    .root_final_group_layout()?,
                );
                let schedule = dp_planned_schedule::<Self>(&key)?;
                setup_matrix_envelope_for_schedule(&schedule)
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

            fn selection_payload_slack_permille() -> u32 {
                $slack_permille
            }

            fn supports_multi_group_final_commit() -> bool {
                <$base>::supports_multi_group_final_commit()
            }

            fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
                $catalog
            }

            fn runtime_schedule(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<akita_types::FoldSchedule, akita_pcs::AkitaError> {
                Self::validate_sis_modulus_profile()?;
                match akita_schedules::resolve_group_batch_schedule(
                    &key,
                    &akita_config::policy_of::<Self>(),
                    Self::ring_challenge_config,
                    Self::fold_challenge_shape_at_level,
                    Self::schedule_catalog(),
                ) {
                    // Catalog misses fall back to the planner DP: the
                    // checked-in table is a performance floor, never a
                    // correctness gate (test-scale shapes sit below the
                    // enumerated production grid).
                    Err(akita_pcs::AkitaError::UnsupportedSchedule(_)) => {
                        dp_planned_schedule::<Self>(&key)
                    }
                    other => other,
                }
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
    crate::schedules::jolt_fp128_d64_onehot_k16_table(),
    // Accept up to 1% larger proofs when that buys a smaller root inner rank
    // `n_a` — the rank multiplies the whole one-hot commit kernel, the
    // dominant prover cost at large T.
    10
);

delegate_preset!(
    /// `D64OneHot` (K=256) with the Jolt-generated large-trace catalog.
    JoltD64OneHotK256,
    akita_config::proof_optimized::fp128::D64OneHot,
    crate::schedules::jolt_fp128_d64_onehot_k256_table(),
    10
);

delegate_preset!(
    /// `D64Dense` with Jolt's DP fallbacks for shapes outside the shipped
    /// catalog (advice/precommitted objects size to their own small shapes).
    /// Zero slack keeps the policy identical to the upstream preset, so the
    /// shipped dense catalog stays identity-valid.
    JoltD64Dense,
    akita_config::proof_optimized::fp128::D64Dense,
    <akita_config::proof_optimized::fp128::D64Dense as CommitmentConfig>::schedule_catalog(),
    0
);

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
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
}
