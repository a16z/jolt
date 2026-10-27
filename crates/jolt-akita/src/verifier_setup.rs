use std::{collections::BTreeSet, sync::Arc};

use akita_config::{
    required_setup_prefix_slot_ids_for_schedule, CommitmentConfig, TrustedScheduleCatalog,
};
use akita_params::{setup_prefix_slot_field_elements, verifier_setup_matrix_capacity_for_schedule};
use akita_pcs::{AkitaError, CpuBackend};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor,
    SetupPrefixVerifierRegistry,
};

use crate::adapters::{AkitaBackendVerifierSetup, AkitaField};

pub(crate) fn streamed_verifier_setup<Cfg: CommitmentConfig<Field = AkitaField>>(
    catalog: &TrustedScheduleCatalog<Cfg>,
    max_num_vars: usize,
    max_total_batch_polys: usize,
) -> Result<AkitaBackendVerifierSetup, AkitaError> {
    let rows = catalog.rows_within_setup_capacity(max_num_vars, max_total_batch_polys)?;
    if rows.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "verifier capacity admits no rows in the requested catalog".to_owned(),
        ));
    }
    let mut retained_fields = 1;
    let mut prefix_ids = BTreeSet::new();
    for row in rows {
        let layout = row.profiles().opening_layout()?;
        retained_fields = retained_fields.max(
            verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout)?
                .num_field_elements,
        );
        prefix_ids.extend(required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &layout,
        )?);
    }
    let mut generation_fields = retained_fields;
    for id in &prefix_ids {
        generation_fields = generation_fields.max(setup_prefix_slot_field_elements(id)?);
    }
    let mut descriptor = AkitaSetupDescriptor {
        max_num_vars,
        max_num_batched_polys: max_total_batch_polys,
        num_field_elements: generation_fields,
        setup_seed: sample_akita_setup_seed(),
    };
    let mut prefixes = SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
    if !prefix_ids.is_empty() {
        let generated = CpuBackend::<AkitaField, AkitaField>::export_setup_prefixes_from_seed(
            &descriptor,
            &prefix_ids.into_iter().collect::<Vec<_>>(),
        )?;
        for slot in generated.verifier_slots() {
            prefixes.insert(slot)?;
        }
    }
    // Prefix commitments stream from the seed; only direct and terminal verifier
    // matrices are retained, sized by Akita's schedule capacity calculation above.
    descriptor.num_field_elements = retained_fields;
    let coefficients = derive_public_matrix_prefix(retained_fields, &descriptor.setup_seed);
    let expanded =
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, coefficients);
    AkitaBackendVerifierSetup::from_parts(Arc::new(expanded), prefixes)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert valid verifier preprocessing"
)]
mod tests {
    use super::*;
    use akita_config::SetupRequirements;
    use akita_pcs::AkitaCommitmentScheme;

    use crate::adapters::with_backend_pool;
    use crate::configs::JoltOneHotK16;
    use crate::{AkitaChunkProfile, AkitaScheduleArtifacts, AKITA_ONE_HOT_K16};

    #[test]
    fn streamed_setup_retains_only_verifier_matrices_and_required_prefixes() {
        let profile = AkitaChunkProfile::default();
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let catalog = TrustedScheduleCatalog::<JoltOneHotK16>::new(
            artifacts
                .one_hot_catalog_for_profile(AKITA_ONE_HOT_K16, profile)
                .unwrap(),
            profile.witness_cfg(),
        )
        .unwrap();
        let requirements = SetupRequirements::from_catalog(&catalog, 25, 51).unwrap();
        assert!(!requirements.prefix_slot_ids().is_empty());
        let setup = with_backend_pool(|| streamed_verifier_setup(&catalog, 25, 51)).unwrap();
        for id in requirements.prefix_slot_ids() {
            assert!(setup.prefix_slots().get(id).is_some());
        }
        let retained = setup.expanded().shared_matrix().num_field_elements();
        assert!(requirements
            .prefix_slot_ids()
            .iter()
            .any(|id| id.n_prefix().unwrap() > retained));
        let scheme = AkitaCommitmentScheme::new(catalog);
        let verifier = with_backend_pool(|| scheme.verifier(setup)).unwrap();
        assert!(scheme
            .schedules()
            .rows_within_setup_capacity(25, 51)
            .unwrap()
            .iter()
            .all(|row| verifier.admits(row.selection().row_digest)));
    }
}
