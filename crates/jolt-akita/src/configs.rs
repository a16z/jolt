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

fn dp_planned_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<akita_types::FoldSchedule, AkitaError> {
    let planned = akita_planner::find_schedule(
        key,
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

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
        let entry_envelope = setup_matrix_envelope_for_schedule(&schedule, Cfg::D)?;
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
    (
        $(#[$doc:meta])*
        $name:ident,
        $base:ty,
        $catalog:expr,
        $basis_range:expr,
        $onehot_chunk_size:expr
    ) => {
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
                let key = AkitaScheduleLookupKey::single(
                    akita_types::OpeningClaimsLayout::new(
                        max_num_vars,
                        max_num_batched_polys,
                    )?
                    .root_final_group_layout()?,
                );
                setup_matrix_envelope_for_schedule(&dp_planned_schedule::<Self>(&key)?, Self::D)
            }

            fn basis_range() -> (u32, u32) {
                $basis_range
            }

            fn onehot_chunk_size() -> usize {
                $onehot_chunk_size
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

            fn runtime_schedule(
                key: AkitaScheduleLookupKey,
            ) -> Result<akita_types::FoldSchedule, AkitaError> {
                Self::validate_sis_modulus_profile()?;
                match akita_schedules::resolve_group_batch_schedule(
                    &key,
                    &akita_config::policy_of::<Self>(),
                    Self::ring_challenge_config,
                    Self::fold_challenge_shape_at_level,
                    Self::schedule_catalog(),
                ) {
                    Err(AkitaError::UnsupportedSchedule(_)) => dp_planned_schedule::<Self>(&key),
                    result => result,
                }
            }

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::FoldSchedule, akita_pcs::AkitaError> {
                if layout.num_groups() == 1 {
                    layout.check()?;
                    Self::runtime_schedule(AkitaScheduleLookupKey::single(
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
    akita_config::proof_optimized::fp128::D64OneHotK16::basis_range(),
    akita_config::proof_optimized::fp128::D64OneHotK16::onehot_chunk_size()
);

delegate_preset!(
    /// `D64OneHot` (K=256) with the Jolt-generated large-trace catalog.
    JoltD64OneHotK256,
    akita_config::proof_optimized::fp128::D64OneHot,
    crate::schedules::jolt_fp128_d64_onehot_k256_table(),
    akita_config::proof_optimized::fp128::D64OneHot::basis_range(),
    akita_config::proof_optimized::fp128::D64OneHot::onehot_chunk_size()
);

delegate_preset!(
    /// D128, K=256 policy for the largest packed trace.
    JoltD128OneHotK256,
    akita_config::proof_optimized::fp128::D128OneHot,
    None,
    (6, 6),
    256
);

delegate_preset!(
    /// `D64Dense` with planner fallback for exact advice and program shapes.
    JoltD64Dense,
    akita_config::proof_optimized::fp128::D64Dense,
    None,
    akita_config::proof_optimized::fp128::D64Dense::basis_range(),
    akita_config::proof_optimized::fp128::D64Dense::onehot_chunk_size()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shapes_have_setup_envelopes() {
        assert!(JoltD64Dense::max_setup_matrix_size(14, 2).is_ok());
        assert!(JoltD64OneHotK16::max_setup_matrix_size(34, 1).is_ok());
        assert!(JoltD64OneHotK256::max_setup_matrix_size(43, 1).is_ok());
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn d128_k256_policy_uses_the_large_trace_geometry() {
        assert_eq!(JoltD128OneHotK256::D, 128);
        assert_eq!(JoltD128OneHotK256::basis_range(), (6, 6));
        assert_eq!(JoltD128OneHotK256::onehot_chunk_size(), 256);

        let layout = akita_types::OpeningClaimsLayout::new(41, 1).unwrap();
        let schedule = JoltD128OneHotK256::get_params_for_prove(&layout).unwrap();
        let commitment = &schedule.root.params.final_group.commitment;
        assert_eq!(commitment.inner_commit_matrix.output_rank(), 4);
        assert_eq!(commitment.num_positions_per_block, 1 << 18);

        let envelope = JoltD128OneHotK256::max_setup_matrix_size(41, 1).unwrap();
        assert_eq!(envelope.max_setup_len * 128 * 16, 11usize << 30);
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn d64_k256_policy_uses_the_min_payload_geometry() {
        let layout = akita_types::OpeningClaimsLayout::new(39, 1).unwrap();
        let schedule = JoltD64OneHotK256::get_params_for_prove(&layout).unwrap();
        let commitment = &schedule.root.params.final_group.commitment;
        assert_eq!(commitment.inner_commit_matrix.output_rank(), 7);
        assert_eq!(commitment.num_positions_per_block, 1 << 20);
    }
}
