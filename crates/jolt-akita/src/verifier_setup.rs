use akita_params::PolynomialGroupLayout;
use jolt_openings::OpeningsError;

use crate::adapters::{
    invalid_setup, validate_one_hot_k, AkitaSetupFlavor, AkitaVerifierScheduleArtifacts,
    BackendVerifierCache,
};
use crate::one_hot_family::OneHotFamily;
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
            backend_cache: BackendVerifierCache::default(),
        })
    }
}
