//! Verifier state shared across setups: one seed-derived matrix, the setup-prefix
//! commitments it needs, and one backend verifier per distinct catalog.
//! Everything is derived from the public seed; nothing is loaded from outside.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use akita_config::required_setup_prefix_slot_ids_for_schedule;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
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

#[derive(Clone)]
struct CatalogDemand {
    setup: AkitaVerifierSetup,
    max_num_vars: usize,
    max_total_batch_polys: usize,
}

impl CatalogDemand {
    fn covers(&self, setup: &AkitaVerifierSetup) -> bool {
        self.setup.same_catalog(setup)
            && setup.max_num_vars <= self.max_num_vars
            && setup.max_total_batch_polys <= self.max_total_batch_polys
    }

    fn merge(demands: &mut Vec<Self>, setup: AkitaVerifierSetup) {
        if let Some(demand) = demands
            .iter_mut()
            .find(|demand| demand.setup.same_catalog(&setup))
        {
            demand.max_num_vars = demand.max_num_vars.max(setup.max_num_vars);
            demand.max_total_batch_polys = demand
                .max_total_batch_polys
                .max(setup.max_total_batch_polys);
        } else {
            demands.push(Self {
                max_num_vars: setup.max_num_vars,
                max_total_batch_polys: setup.max_total_batch_polys,
                setup,
            });
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AkitaVerifierRequirements {
    pub(crate) max_num_vars: usize,
    pub(crate) max_total_batch_polys: usize,
    pub(crate) retained_field_elements: usize,
    pub(crate) generation_field_elements: usize,
    pub(crate) prefix_slots: BTreeSet<SetupPrefixSlotId>,
}

impl AkitaVerifierRequirements {
    #[cfg(test)]
    pub(crate) fn for_setups(setups: &[AkitaVerifierSetup]) -> Result<Self, OpeningsError> {
        let mut demands = Vec::new();
        for setup in setups {
            CatalogDemand::merge(&mut demands, setup.clone());
        }
        Self::for_demands(&demands)
    }

    fn for_demands(demands: &[CatalogDemand]) -> Result<Self, OpeningsError> {
        if demands.is_empty() {
            return Err(invalid_setup("verifier provisioning requires a catalog"));
        }
        let mut requirements = Self {
            max_num_vars: 0,
            max_total_batch_polys: 0,
            retained_field_elements: 1,
            generation_field_elements: 1,
            prefix_slots: BTreeSet::new(),
        };
        for demand in demands {
            let setup = &demand.setup;
            let mut max_num_vars = demand.max_num_vars;
            if setup.schedule_artifacts.one_hot().is_some() {
                max_num_vars = max_num_vars.max(setup.one_hot_backend_num_vars()?);
            }
            let batch = demand.max_total_batch_polys;
            requirements.max_num_vars = requirements.max_num_vars.max(max_num_vars);
            requirements.max_total_batch_polys = requirements.max_total_batch_polys.max(batch);
            if setup.schedule_artifacts.dense().is_some() {
                requirements.include(setup.dense_scheme()?.schedules(), max_num_vars, batch)?;
            }
            if setup.schedule_artifacts.one_hot().is_some() {
                with_one_hot_family!(scheme setup.one_hot_scheme()?, |scheme| {
                    requirements.include(scheme.schedules(), max_num_vars, batch)
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
        max_num_vars: usize,
        max_total_batch_polys: usize,
    ) -> Result<(), OpeningsError> {
        let rows = catalog
            .rows_within_setup_capacity(max_num_vars, max_total_batch_polys)
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
            for id in required_setup_prefix_slot_ids_for_schedule(row.schedule(), &layout)
                .map_err(invalid_setup)?
            {
                self.generation_field_elements = self
                    .generation_field_elements
                    .max(setup_prefix_slot_field_elements(&id).map_err(invalid_setup)?);
                let _ = self.prefix_slots.insert(id);
            }
        }
        Ok(())
    }
}

/// Derive the matrix and prefix commitments, reusing what `previous` already holds.
/// Only missing prefix slots are computed. The generation matrix is truncated to
/// the verifier's need afterwards.
fn generate_material(
    requirements: &AkitaVerifierRequirements,
    previous: Option<&AkitaBackendVerifierSetup>,
) -> Result<AkitaBackendVerifierSetup, OpeningsError> {
    let seed = sample_akita_setup_seed();
    let mut prefixes = previous.map_or_else(
        || SetupPrefixVerifierRegistry::new(seed.clone()),
        |previous| previous.prefix_slots().clone(),
    );
    let missing = requirements
        .prefix_slots
        .iter()
        .filter(|id| prefixes.get(id).is_none())
        .cloned()
        .collect::<Vec<_>>();
    let covers = |setup: &AkitaBackendVerifierSetup, fields: usize| {
        let descriptor = setup.expanded().descriptor();
        descriptor.num_field_elements >= fields
            && descriptor.max_num_vars >= requirements.max_num_vars
            && descriptor.max_num_batched_polys >= requirements.max_total_batch_polys
    };
    let retained_fields =
        requirements
            .retained_field_elements
            .max(previous.map_or(0, |previous| {
                previous.expanded().descriptor().num_field_elements
            }));
    let mut generation_fields = retained_fields;
    for id in &missing {
        generation_fields =
            generation_fields.max(setup_prefix_slot_field_elements(id).map_err(invalid_setup)?);
    }
    let descriptor = |num_field_elements| AkitaSetupDescriptor {
        max_num_vars: requirements.max_num_vars,
        max_num_batched_polys: requirements.max_total_batch_polys,
        num_field_elements,
        setup_seed: seed.clone(),
    };
    let expanded = match previous {
        Some(previous) if covers(previous, generation_fields) => Arc::clone(previous.expanded()),
        previous => {
            // Only the capacity grew: the existing coefficients are already the full prefix.
            let coefficients = previous
                .filter(|previous| {
                    previous.expanded().descriptor().num_field_elements >= generation_fields
                })
                .map_or_else(
                    || derive_public_matrix_prefix(generation_fields, &seed),
                    |previous| previous.expanded().shared_matrix().clone(),
                );
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor(generation_fields),
                    coefficients,
                ),
            )
        }
    };
    if !missing.is_empty() {
        let backend = CpuBackend::<AkitaField, AkitaField>::new(Arc::clone(&expanded))
            .map_err(invalid_setup)?;
        let generated = backend
            .export_setup_prefixes(&missing)
            .map_err(invalid_setup)?;
        for slot in generated.verifier_slots() {
            prefixes.insert(slot).map_err(invalid_setup)?;
        }
    }
    let expanded = match previous.filter(|previous| covers(previous, retained_fields)) {
        Some(previous) => Arc::clone(previous.expanded()),
        None if expanded.descriptor().num_field_elements > retained_fields => {
            let coefficients = expanded
                .shared_matrix()
                .as_field_slice()
                .get(..retained_fields)
                .ok_or_else(|| invalid_setup("generated verifier matrix is too short"))?;
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor(retained_fields),
                    FlatMatrix::from_flat_data(coefficients.to_vec()),
                ),
            )
        }
        None => expanded,
    };
    AkitaBackendVerifierSetup::from_parts(expanded, prefixes).map_err(invalid_setup)
}

pub(crate) struct Prepared {
    material: AkitaBackendVerifierSetup,
    catalogs: Vec<CatalogDemand>,
}

impl Prepared {
    fn grow(
        setups: impl IntoIterator<Item = Result<AkitaVerifierSetup, OpeningsError>>,
        previous: Option<&Self>,
    ) -> Result<Self, OpeningsError> {
        let mut demands = previous.map_or_else(Vec::new, |previous| previous.catalogs.clone());
        for setup in setups {
            CatalogDemand::merge(&mut demands, setup?);
        }
        let requirements = AkitaVerifierRequirements::for_demands(&demands)?;
        let material = with_backend_pool(|| {
            generate_material(&requirements, previous.map(|previous| &previous.material))
        })?;
        let reusable = previous
            .filter(|previous| Arc::ptr_eq(previous.material.expanded(), material.expanded()));
        for demand in &mut demands {
            // A verifier holds the prefix registry it was built with, so it is kept
            // only when its catalog's demand did not grow past what it was built for.
            let existing = reusable.and_then(|previous| {
                previous.catalogs.iter().find(|other| {
                    demand.setup.same_catalog(&other.setup)
                        && demand.max_num_vars <= other.max_num_vars
                        && demand.max_total_batch_polys <= other.max_total_batch_polys
                })
            });
            if let Some(existing) = existing {
                demand.setup = existing.setup.clone();
                continue;
            }
            demand.setup.backend_cache = demand.setup.backend_cache.decoded_only();
            let artifacts = &demand.setup.schedule_artifacts;
            let dense = artifacts.dense().map(|_| material.clone());
            let one_hot = artifacts.one_hot().map(|_| material.clone());
            demand.setup.prime_backend_cache(dense, one_hot)?;
        }
        Ok(Self {
            material,
            catalogs: demands,
        })
    }

    fn size(&self) -> usize {
        self.material.expanded().descriptor().num_field_elements
    }

    fn catalog(&self, setup: &AkitaVerifierSetup) -> Result<&AkitaVerifierSetup, OpeningsError> {
        if setup.max_num_polys_per_commitment_group == 0
            || setup.max_num_polys_per_commitment_group > setup.max_total_batch_polys
        {
            return Err(invalid_setup(
                "verifier group width must fit its batch capacity",
            ));
        }
        self.catalogs
            .iter()
            .find(|demand| demand.covers(setup))
            .map(|demand| &demand.setup)
            .ok_or_else(|| invalid_setup("verifier catalog or capacity was not prepared"))
    }
}

/// Verifier state prepared ahead of proofs. While a value is alive, any setup of
/// a covered catalog and capacity attaches to it on first use instead of building
/// its own; uncovered setups grow a new shared state on their first proof.
#[derive(Clone)]
pub struct AkitaVerifierResources {
    /// Holds the published state; the registry only keeps a weak reference.
    _prepared: Arc<Prepared>,
}

impl AkitaVerifierResources {
    /// Decode and audit each distinct catalog, derive one matrix from the public
    /// seed at the largest capacity any setup needs, compute the required prefix
    /// commitments and build each catalog's backend verifier.
    /// Fails on the first `Err` before generating or publishing anything.
    pub fn prepare(
        setups: impl IntoIterator<Item = Result<AkitaVerifierSetup, OpeningsError>>,
    ) -> Result<Self, OpeningsError> {
        let prepared = Arc::new(Prepared::grow(setups, None)?);
        LAZY_VERIFIER_RESOURCES.publish(&prepared);
        Ok(Self {
            _prepared: prepared,
        })
    }
}

pub(crate) struct LazyVerifierResources {
    /// Every live prepared state. A grown state never hides an older one that is
    /// still owned, so a setup it covers keeps attaching to it.
    live: Mutex<Vec<Weak<Prepared>>>,
    /// Serializes growth only: lookups never wait for it, and a miss that waited
    /// re-checks so concurrent misses for one catalog build it once.
    growing: Mutex<()>,
}

impl LazyVerifierResources {
    pub(crate) const fn new() -> Self {
        Self {
            live: Mutex::new(Vec::new()),
            growing: Mutex::new(()),
        }
    }

    fn live(&self) -> MutexGuard<'_, Vec<Weak<Prepared>>> {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live.retain(|prepared| prepared.strong_count() > 0);
        live
    }

    fn publish(&self, prepared: &Arc<Prepared>) {
        self.live().push(Arc::downgrade(prepared));
    }

    /// A live prepared catalog with exactly `setup`'s bytes, already decoded and audited.
    fn prepared_catalog<T>(
        &self,
        setup: &AkitaVerifierSetup,
        decoded: impl Fn(&BackendVerifierCache) -> Option<T>,
    ) -> Option<T> {
        self.live()
            .iter()
            .filter_map(Weak::upgrade)
            .find_map(|state| {
                state
                    .catalogs
                    .iter()
                    .find(|demand| demand.setup.same_catalog(setup))
                    .and_then(|demand| decoded(&demand.setup.backend_cache))
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

    pub(crate) fn attach(&self, setup: &AkitaVerifierSetup) -> Result<(), OpeningsError> {
        let prepared = match self.covering(setup) {
            Some(prepared) => prepared,
            None => self.grow_for(setup)?,
        };
        let catalog = prepared.catalog(setup)?;
        setup
            .backend_cache
            .adopt(&catalog.backend_cache, Arc::clone(&prepared));
        Ok(())
    }

    fn grow_for(&self, setup: &AkitaVerifierSetup) -> Result<Arc<Prepared>, OpeningsError> {
        let _growing = self.growing.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(prepared) = self.covering(setup) {
            return Ok(prepared);
        }
        let largest = self
            .live()
            .iter()
            .filter_map(Weak::upgrade)
            .max_by_key(|state| state.size());
        let grown = Arc::new(Prepared::grow([Ok(setup.clone())], largest.as_deref())?);
        self.publish(&grown);
        Ok(grown)
    }

    /// The largest live state that covers `setup`.
    fn covering(&self, setup: &AkitaVerifierSetup) -> Option<Arc<Prepared>> {
        self.live()
            .iter()
            .filter_map(Weak::upgrade)
            .filter(|state| state.catalog(setup).is_ok())
            .max_by_key(|state| state.size())
    }
}

pub(crate) static LAZY_VERIFIER_RESOURCES: LazyVerifierResources = LazyVerifierResources::new();

impl AkitaVerifierSetup {
    pub(crate) fn same_catalog(&self, other: &Self) -> bool {
        self.one_hot_k == other.one_hot_k && self.schedule_artifacts == other.schedule_artifacts
    }
}

#[cfg(test)]
mod tests;
