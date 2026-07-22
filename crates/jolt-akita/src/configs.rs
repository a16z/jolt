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

use akita_config::{setup_level_params_from_schedule, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::GeneratedScheduleTable;
use akita_types::{AkitaScheduleLookupKey, LevelParams, SetupMatrixEnvelope, Step};

fn include_matrix(
    max_setup_len: &mut usize,
    rows: usize,
    columns: usize,
    role: &str,
) -> Result<(), AkitaError> {
    let len = rows
        .checked_mul(columns)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} setup envelope overflow")))?;
    *max_setup_len = (*max_setup_len).max(len);
    Ok(())
}

fn include_level(params: &LevelParams, max_setup_len: &mut usize) -> Result<(), AkitaError> {
    if !params.precommitted_groups.is_empty() || params.setup_prefix.is_some() {
        return Err(AkitaError::InvalidSetup(
            "Jolt's scalar schedule catalog contains multi-group setup metadata".to_string(),
        ));
    }
    include_matrix(
        max_setup_len,
        params.a_key.row_len(),
        params.inner_width(),
        "A",
    )?;
    include_matrix(
        max_setup_len,
        params.b_key.row_len(),
        params.outer_width(),
        "B",
    )?;
    include_matrix(
        max_setup_len,
        params.d_key.row_len(),
        params.d_matrix_width(),
        "D",
    )
}

fn include_scalar_root(
    params: &LevelParams,
    num_polynomials: usize,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let d_width = num_polynomials
        .checked_mul(params.num_blocks)
        .and_then(|width| width.checked_mul(params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("root D setup width overflow".to_string()))?;
    let b_width = params
        .a_key
        .row_len()
        .checked_mul(params.num_digits_open)
        .and_then(|width| width.checked_mul(params.num_blocks))
        .and_then(|width| width.checked_mul(num_polynomials))
        .ok_or_else(|| AkitaError::InvalidSetup("root B setup width overflow".to_string()))?;
    include_matrix(
        max_setup_len,
        params.a_key.row_len(),
        params.inner_width(),
        "root A",
    )?;
    include_matrix(max_setup_len, params.b_key.row_len(), b_width, "root B")?;
    include_matrix(max_setup_len, params.d_key.row_len(), d_width, "root D")
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
        entry.precommitteds.is_empty()
            && entry.final_group.num_vars() == max_num_vars
            && entry.final_group.num_polynomials() == max_num_batched_polys
    });
    if !requested_shape_is_catalogued {
        return Ok(None);
    }

    let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
    for entry in table.entries.iter().filter(|entry| {
        entry.precommitteds.is_empty()
            && entry.final_group.num_vars() <= max_num_vars
            && entry.final_group.num_polynomials() <= max_num_batched_polys
    }) {
        let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(entry.final_group))?;
        for params in setup_level_params_from_schedule(&schedule) {
            include_level(&params, &mut envelope.max_setup_len)?;
        }
        let root_params = match schedule.steps.first() {
            Some(Step::Fold(step)) => Some(&step.params),
            Some(Step::Direct(step)) => step.params.as_ref(),
            None => {
                return Err(AkitaError::InvalidSetup(
                    "Jolt catalog schedule has no steps".to_string(),
                ));
            }
        };
        if let Some(params) = root_params {
            include_scalar_root(
                params,
                entry.final_group.num_polynomials(),
                &mut envelope.max_setup_len,
            )?;
        }
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
