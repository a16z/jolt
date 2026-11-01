use akita_params::PolynomialGroupLayout;
use jolt_openings::OpeningsError;

use crate::adapters::{
    invalid_setup, validate_one_hot_k, AkitaSetupFlavor, AkitaVerifierScheduleArtifacts,
    BackendVerifierCache,
};
use crate::one_hot_family::OneHotFamily;
use crate::verifier_resources::ExpandedPrefixSource;
use crate::{AkitaChunkProfile, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup};

impl AkitaScheme {
    /// Build shape-specific public metadata without generating matrices or commitments.
    pub fn verifier_setup_from_params(
        params: &AkitaSetupParams,
    ) -> Result<AkitaVerifierSetup, OpeningsError> {
        if params.max_num_polys_per_commitment_group == 0
            || params.max_num_polys_per_commitment_group > params.max_total_batch_polys
        {
            return Err(invalid_setup(
                "verifier group width must fit its nonzero batch capacity",
            ));
        }
        if params.flavor == AkitaSetupFlavor::Dense
            && params.akita_chunk_profile != AkitaChunkProfile::default()
        {
            return Err(OpeningsError::InvalidSetup(
                "dense-only Akita setup cannot select one-hot witness chunking".to_owned(),
            ));
        }
        if params.flavor != AkitaSetupFlavor::Dense {
            OneHotFamily::from_parts(params.one_hot_k, params.akita_chunk_profile)
                .and_then(|family| {
                    family.validate_num_vars(
                        params.max_num_vars,
                        params.max_num_polys_per_commitment_group,
                    )
                })
                .map_err(invalid_setup)?;
        }
        if params.grouped_schedule.as_ref().is_some_and(|request| {
            request.final_group()
                != PolynomialGroupLayout::new(
                    params.max_num_vars,
                    params.max_num_polys_per_commitment_group,
                )
        }) {
            return Err(OpeningsError::InvalidSetup(
                "the grouped schedule request final shape must equal setup shape".to_owned(),
            ));
        }
        let _ = validate_one_hot_k(params.one_hot_k).map_err(invalid_setup)?;
        let artifacts = &params.schedule_artifacts;
        let dense_catalog = artifacts.dense_catalog().map_err(invalid_setup)?;
        let dense_schedule_artifact = || dense_catalog.to_artifact_bytes().map_err(invalid_setup);
        let one_hot_schedule_artifact = || {
            let base = artifacts
                .one_hot_catalog_for_profile(params.one_hot_k, params.akita_chunk_profile)
                .map_err(invalid_setup)?;
            let catalog = params
                .grouped_schedule
                .as_ref()
                .map_or_else(
                    || Ok(base.clone()),
                    |grouped_schedule| {
                        grouped_schedule.extend_catalog(
                            &dense_catalog,
                            &artifacts.full_dense_catalog()?,
                            &base,
                            params.one_hot_k,
                            params.akita_chunk_profile,
                        )
                    },
                )
                .map_err(invalid_setup)?;
            catalog.to_artifact_bytes().map_err(invalid_setup)
        };
        let profile = params.akita_chunk_profile;
        let schedule_artifacts = match params.flavor {
            AkitaSetupFlavor::Both => AkitaVerifierScheduleArtifacts::Both {
                dense_profile: artifacts.dense_profile(),
                profile,
                dense: dense_schedule_artifact()?,
                one_hot: one_hot_schedule_artifact()?,
            },
            AkitaSetupFlavor::OneHot => AkitaVerifierScheduleArtifacts::OneHot {
                profile,
                one_hot: one_hot_schedule_artifact()?,
            },
            AkitaSetupFlavor::Dense => AkitaVerifierScheduleArtifacts::Dense {
                dense_profile: artifacts.dense_profile(),
                dense: dense_schedule_artifact()?,
            },
        };
        Ok(AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            max_total_batch_polys: params.max_total_batch_polys,
            default_layout_digest: params.default_layout_digest,
            one_hot_k: params.one_hot_k,
            schedule_artifacts: schedule_artifacts.into(),
            prefix_commitments: None,
            admitted_shapes: Vec::new(),
            backend_cache: BackendVerifierCache::default(),
        })
    }
}

/// Preprocessing: one verifier setup that verifies every shape of `setups`.
/// Their one-hot catalogs merge into one, so the result stores a single catalog
/// and key load prepares a single backend key; its own fields describe the
/// largest shape, and it stores the setup-prefix commitments every shape needs,
/// committing to those no input stores from `expanded` where it holds them.
/// All of `setups` must share their dense catalog and one-hot family.
pub(crate) fn merge_verifier_setups(
    setups: &[AkitaVerifierSetup],
    expanded: Option<ExpandedPrefixSource<'_>>,
) -> Result<AkitaVerifierSetup, OpeningsError> {
    let largest = setups
        .iter()
        .max_by_key(|setup| setup.shape())
        .ok_or_else(|| invalid_setup("no verifier setups to merge"))?;
    if setups.iter().any(|setup| {
        setup.one_hot_k != largest.one_hot_k
            || setup.akita_chunk_profile() != largest.akita_chunk_profile()
            || setup.schedule_artifacts.dense() != largest.schedule_artifacts.dense()
            || setup.schedule_artifacts.one_hot().is_some()
                != largest.schedule_artifacts.one_hot().is_some()
    }) {
        return Err(invalid_setup(
            "merged verifier setups must share their dense catalog and one-hot family",
        ));
    }
    let mut shapes = setups
        .iter()
        .flat_map(|setup| {
            if setup.admitted_shapes.is_empty() {
                vec![setup.shape()]
            } else {
                setup.admitted_shapes.clone()
            }
        })
        .collect::<Vec<_>>();
    shapes.sort();
    shapes.dedup();
    let mut merged = AkitaVerifierSetup {
        prefix_commitments: None,
        admitted_shapes: if shapes.len() > 1 { shapes } else { Vec::new() },
        backend_cache: BackendVerifierCache::default(),
        ..largest.clone()
    };
    if largest.schedule_artifacts.one_hot().is_some() {
        let catalogs = setups
            .iter()
            .filter_map(|setup| setup.schedule_artifacts.one_hot())
            .collect::<Vec<_>>();
        let union = crate::schedule_registry::union_one_hot_catalogs(
            largest.one_hot_k,
            largest.akita_chunk_profile(),
            &catalogs,
        )
        .map_err(invalid_setup)?;
        merged.schedule_artifacts = largest.schedule_artifacts.with_one_hot(union).into();
    }
    let known = crate::verifier_resources::stored_prefixes(setups)?;
    crate::verifier_resources::provision_prefix_commitments(
        std::slice::from_mut(&mut merged),
        &known,
        expanded,
    )?;
    Ok(merged)
}
