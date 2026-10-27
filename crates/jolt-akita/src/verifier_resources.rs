//! Verifier state shared across setups: one seed-derived matrix, the setup-prefix
//! commitments it needs, and one backend verifier per distinct catalog.
//! Everything is derived from the public seed; nothing is loaded from outside.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use akita_config::{CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_params::{
    setup_prefix_slot_field_elements, verifier_setup_matrix_capacity_for_schedule, FlatMatrix,
    SetupPrefixSlotId,
};
use akita_pcs::CpuBackend;
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor,
    SetupPrefixVerifierRegistry,
};
use jolt_openings::OpeningsError;

use crate::adapters::{
    invalid_setup, with_backend_pool, AkitaBackendScheme, AkitaBackendVerifierSetup,
    BackendVerifierCache,
};
use crate::one_hot_family::with_one_hot_family;
use crate::one_hot_family::AkitaOneHotBackendScheme;
use crate::{AkitaField, AkitaVerifierSetup};

#[derive(Clone, Debug)]
pub(crate) struct AkitaVerifierRequirements {
    pub(crate) max_num_vars: usize,
    pub(crate) max_total_batch_polys: usize,
    pub(crate) retained_field_elements: usize,
    pub(crate) generation_field_elements: usize,
    pub(crate) prefix_slots: BTreeSet<SetupPrefixSlotId>,
}

impl AkitaVerifierRequirements {
    /// One state's capacity, the per-field maximum over `setups` including each
    /// one-hot catalog's backend arity, and what every distinct catalog needs at it.
    pub(crate) fn for_setups(setups: &[AkitaVerifierSetup]) -> Result<Self, OpeningsError> {
        if setups.is_empty() {
            return Err(invalid_setup("verifier provisioning requires a catalog"));
        }
        let mut requirements = Self {
            max_num_vars: 0,
            max_total_batch_polys: 0,
            retained_field_elements: 1,
            generation_field_elements: 1,
            prefix_slots: BTreeSet::new(),
        };
        for setup in setups {
            let mut max_num_vars = setup.max_num_vars;
            if setup.schedule_artifacts.one_hot().is_some() {
                max_num_vars = max_num_vars.max(setup.one_hot_backend_num_vars()?);
            }
            requirements.max_num_vars = requirements.max_num_vars.max(max_num_vars);
            requirements.max_total_batch_polys = requirements
                .max_total_batch_polys
                .max(setup.max_total_batch_polys);
        }
        // Every catalog's verifier admits rows up to the state's descriptor, so each
        // is sized at that capacity and gets the prefix slots of every row it admits.
        for (index, setup) in setups.iter().enumerate() {
            if setups
                .iter()
                .take(index)
                .any(|earlier| earlier.same_catalog(setup))
            {
                continue;
            }
            if setup.schedule_artifacts.dense().is_some() {
                requirements.include(setup.dense_scheme()?.schedules())?;
            }
            if setup.schedule_artifacts.one_hot().is_some() {
                with_one_hot_family!(scheme setup.one_hot_scheme()?, |scheme| {
                    requirements.include(scheme.schedules())
                })?;
            }
        }
        requirements.generation_field_elements = requirements
            .generation_field_elements
            .max(requirements.retained_field_elements);
        Ok(requirements)
    }

    fn include<Cfg: CommitmentConfig<Field = AkitaField>>(
        &mut self,
        catalog: &TrustedScheduleCatalog<Cfg>,
    ) -> Result<(), OpeningsError> {
        let rows = catalog
            .rows_within_setup_capacity(self.max_num_vars, self.max_total_batch_polys)
            .map_err(invalid_setup)?;
        if rows.is_empty() {
            return Err(invalid_setup(
                "verifier capacity admits no rows in a requested catalog",
            ));
        }
        for row in rows {
            let layout = row.profiles().opening_layout().map_err(invalid_setup)?;
            let capacity = verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout)
                .map_err(invalid_setup)?;
            self.retained_field_elements = self
                .retained_field_elements
                .max(capacity.num_field_elements);
        }
        let akita =
            SetupRequirements::from_catalog(catalog, self.max_num_vars, self.max_total_batch_polys)
                .map_err(invalid_setup)?;
        for id in akita.prefix_slot_ids() {
            self.generation_field_elements = self
                .generation_field_elements
                .max(setup_prefix_slot_field_elements(id).map_err(invalid_setup)?);
            let _ = self.prefix_slots.insert(id.clone());
        }
        Ok(())
    }
}

/// Derive the matrix and the required prefix commitments from the public seed.
/// The generation matrix is truncated to the verifier's need afterwards.
fn generate_material(
    requirements: &AkitaVerifierRequirements,
) -> Result<AkitaBackendVerifierSetup, OpeningsError> {
    let seed = sample_akita_setup_seed();
    let descriptor = |num_field_elements| AkitaSetupDescriptor {
        max_num_vars: requirements.max_num_vars,
        max_num_batched_polys: requirements.max_total_batch_polys,
        num_field_elements,
        setup_seed: seed.clone(),
    };
    let generation_fields = requirements.generation_field_elements;
    let retained_fields = requirements.retained_field_elements;
    let mut expanded = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            descriptor(generation_fields),
            derive_public_matrix_prefix(generation_fields, &seed),
        ),
    );
    let mut prefixes = SetupPrefixVerifierRegistry::new(seed.clone());
    if !requirements.prefix_slots.is_empty() {
        let slots = requirements
            .prefix_slots
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let backend = CpuBackend::<AkitaField, AkitaField>::new(Arc::clone(&expanded))
            .map_err(invalid_setup)?;
        let generated = backend
            .export_setup_prefixes(&slots)
            .map_err(invalid_setup)?;
        for slot in generated.verifier_slots() {
            prefixes.insert(slot).map_err(invalid_setup)?;
        }
    }
    if generation_fields > retained_fields {
        let coefficients = expanded
            .shared_matrix()
            .as_field_slice()
            .get(..retained_fields)
            .ok_or_else(|| invalid_setup("generated verifier matrix is too short"))?;
        expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                descriptor(retained_fields),
                FlatMatrix::from_flat_data(coefficients.to_vec()),
            ),
        );
    }
    AkitaBackendVerifierSetup::from_parts(expanded, prefixes).map_err(invalid_setup)
}

pub(crate) struct Prepared {
    material: AkitaBackendVerifierSetup,
    /// One setup per distinct catalog, holding that catalog's backend verifiers.
    catalogs: Vec<AkitaVerifierSetup>,
}

impl Prepared {
    fn new(
        setups: impl IntoIterator<Item = Result<AkitaVerifierSetup, OpeningsError>>,
    ) -> Result<Self, OpeningsError> {
        let mut requested = Vec::new();
        for setup in setups {
            let setup = setup?;
            setup.check_group_width()?;
            requested.push(setup);
        }
        let requirements = AkitaVerifierRequirements::for_setups(&requested)?;
        let material = with_backend_pool(|| generate_material(&requirements))?;
        let mut catalogs: Vec<AkitaVerifierSetup> = Vec::new();
        for mut setup in requested {
            if catalogs.iter().any(|catalog| catalog.same_catalog(&setup)) {
                continue;
            }
            setup.backend_cache = setup.backend_cache.decoded_only();
            let artifacts = &setup.schedule_artifacts;
            let dense = artifacts.dense().map(|_| material.clone());
            let one_hot = artifacts.one_hot().map(|_| material.clone());
            setup.prime_backend_cache(dense, one_hot)?;
            catalogs.push(setup);
        }
        Ok(Self { material, catalogs })
    }

    fn size(&self) -> usize {
        self.material.expanded().descriptor().num_field_elements
    }

    /// The prepared catalog serving `setup`: identical catalog bytes, within the
    /// capacity every catalog of this state is provisioned at.
    fn catalog(&self, setup: &AkitaVerifierSetup) -> Option<&AkitaVerifierSetup> {
        let descriptor = self.material.expanded().descriptor();
        if setup.max_num_vars > descriptor.max_num_vars
            || setup.max_total_batch_polys > descriptor.max_num_batched_polys
        {
            return None;
        }
        self.catalogs
            .iter()
            .find(|catalog| catalog.same_catalog(setup))
    }
}

/// Verifier state prepared ahead of proofs. While a value is alive, a setup with
/// a prepared catalog's bytes that fits the state's capacity attaches to it on
/// first use instead of building its own. Dropping every value stops new setups
/// from attaching; setups already attached keep their verifiers.
#[derive(Clone)]
#[must_use = "setups attach to the prepared state only while it is alive"]
pub struct AkitaVerifierResources {
    /// Holds the published state; the registry only keeps a weak reference.
    _prepared: Arc<Prepared>,
}

impl AkitaVerifierResources {
    /// Decode and audit each distinct catalog, derive one matrix from the public
    /// seed, compute the required prefix commitments and build each catalog's
    /// backend verifier. Fails on the first `Err` before generating or publishing
    /// anything.
    ///
    /// Setups prepared together share one state at the per-field maximum of their
    /// capacities, including each one-hot catalog's backend arity, and every
    /// catalog in it is provisioned at that maximum. Prepare
    /// very different shapes separately: together they can cost more than apart.
    pub fn prepare(
        setups: impl IntoIterator<Item = Result<AkitaVerifierSetup, OpeningsError>>,
    ) -> Result<Self, OpeningsError> {
        let prepared = Arc::new(Prepared::new(setups)?);
        PREPARED_VERIFIER_RESOURCES.publish(&prepared);
        Ok(Self {
            _prepared: prepared,
        })
    }
}

/// Live prepared states, held weakly so each dies with its last handle.
pub(crate) struct PreparedRegistry {
    live: Mutex<Vec<Weak<Prepared>>>,
}

impl PreparedRegistry {
    pub(crate) const fn new() -> Self {
        Self {
            live: Mutex::new(Vec::new()),
        }
    }

    /// The live states, newest first. Matching runs after the lock is released:
    /// comparing catalogs reads megabytes.
    fn states(&self) -> Vec<Arc<Prepared>> {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live.retain(|prepared| prepared.strong_count() > 0);
        live.iter().rev().filter_map(Weak::upgrade).collect()
    }

    fn publish(&self, prepared: &Arc<Prepared>) {
        self.live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::downgrade(prepared));
    }

    /// A live prepared catalog with exactly `setup`'s bytes, already decoded and audited.
    fn prepared_catalog<T>(
        &self,
        setup: &AkitaVerifierSetup,
        decoded: impl Fn(&BackendVerifierCache) -> Option<T>,
    ) -> Option<T> {
        self.states().iter().find_map(|state| {
            state
                .catalogs
                .iter()
                .find(|catalog| catalog.same_catalog(setup))
                .and_then(|catalog| decoded(&catalog.backend_cache))
        })
    }

    pub(crate) fn decoded_dense(
        &self,
        setup: &AkitaVerifierSetup,
    ) -> Option<Arc<AkitaBackendScheme>> {
        self.prepared_catalog(setup, BackendVerifierCache::decoded_dense)
    }

    pub(crate) fn decoded_one_hot(
        &self,
        setup: &AkitaVerifierSetup,
    ) -> Option<Arc<AkitaOneHotBackendScheme>> {
        self.prepared_catalog(setup, BackendVerifierCache::decoded_one_hot)
    }

    /// Attach `setup` to the smallest live state that covers it, the newest among
    /// equals. Otherwise build state for it alone; it is not published, and its
    /// verifiers live in `setup` and its clones. No lock is held while building: the build runs on
    /// the backend's rayon pool, which may run another verification on this
    /// thread while it waits.
    pub(crate) fn attach(&self, setup: &AkitaVerifierSetup) -> Result<(), OpeningsError> {
        setup.check_group_width()?;
        let states = self.states();
        let covering = states
            .iter()
            .filter_map(|state| state.catalog(setup).map(|catalog| (state.size(), catalog)))
            .min_by_key(|(size, _)| *size);
        let own;
        let catalog = if let Some((_, catalog)) = covering {
            catalog
        } else {
            own = Prepared::new([Ok(setup.clone())])?;
            own.catalog(setup)
                .ok_or_else(|| invalid_setup("verifier state does not cover its setup"))?
        };
        setup.backend_cache.adopt(&catalog.backend_cache);
        Ok(())
    }
}

pub(crate) static PREPARED_VERIFIER_RESOURCES: PreparedRegistry = PreparedRegistry::new();

impl AkitaVerifierSetup {
    pub(crate) fn same_catalog(&self, other: &Self) -> bool {
        self.one_hot_k == other.one_hot_k
            && (Arc::ptr_eq(&self.schedule_artifacts, &other.schedule_artifacts)
                || self.schedule_artifacts == other.schedule_artifacts)
    }

    fn check_group_width(&self) -> Result<(), OpeningsError> {
        if self.max_num_polys_per_commitment_group == 0
            || self.max_num_polys_per_commitment_group > self.max_total_batch_polys
        {
            return Err(invalid_setup(
                "verifier group width must fit its batch capacity",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
