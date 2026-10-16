//! Setup-owned grouped schedule catalog construction.
//!
//! Base scalar rows come from checked-in external artifacts. Program-specific
//! advice and committed-program shapes are guided from the approved scalar row
//! during preprocessing. Chunked advice or a field increment with optional
//! advice may require a new grouped schedule when that scalar row's fixed
//! recursion geometry is infeasible. Every planned row is audited and merged
//! into the setup's immutable catalog; runtime proving and verification never
//! plan schedules.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
    ScheduleRowDigest,
};
use akita_pcs::AkitaError;
use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
use akita_planner::find_adapted_schedule;
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use serde::{Deserialize, Serialize};

use crate::configs::{AkitaChunkProfile, JoltDenseBounded, JoltDenseFull};
use crate::one_hot_family::{with_one_hot_family, OneHotFamily, AKITA_ONE_HOT_K256};

const MAX_PROVISIONED_ROWS: usize = 128;

/// Physical shape and admitted coefficient range of one dense commitment group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenseGroupLayout {
    Bounded { num_vars: usize },
    FullWidth { num_vars: usize },
}

impl DenseGroupLayout {
    fn producer(
        self,
        bounded: &ValidatedScheduleCatalog,
        full_width: &ValidatedScheduleCatalog,
    ) -> Result<PrecommittedProducer, AkitaError> {
        match self {
            Self::Bounded { num_vars } => producer::<JoltDenseBounded>(&dense_group_profile(
                bounded,
                PolynomialGroupLayout::new(num_vars, 1),
            )?),
            Self::FullWidth { num_vars } => producer::<JoltDenseFull>(&dense_group_profile(
                full_width,
                PolynomialGroupLayout::new(num_vars, 1),
            )?),
        }
    }
}

fn producer<Cfg: CommitmentConfig>(
    profile: &GroupCommitPhaseParams,
) -> Result<PrecommittedProducer, AkitaError> {
    PrecommittedProducer::try_new(*profile, Cfg::committed_source_contract()?)
}

/// Public inputs needed to construct this setup's grouped schedules.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedScheduleParams {
    untrusted_physical_arity: Option<usize>,
    trusted_physical_arity: Option<usize>,
    #[serde(default)]
    mandatory_dense_layouts: Vec<DenseGroupLayout>,
    final_group: PolynomialGroupLayout,
}

impl GroupedScheduleParams {
    pub fn new(
        untrusted_physical_num_vars: Option<usize>,
        trusted_physical_num_vars: Option<usize>,
        mandatory_dense_layouts: Vec<DenseGroupLayout>,
        final_group: PolynomialGroupLayout,
    ) -> Self {
        Self {
            untrusted_physical_arity: untrusted_physical_num_vars,
            trusted_physical_arity: trusted_physical_num_vars,
            mandatory_dense_layouts,
            final_group,
        }
    }

    pub(crate) fn full_width_arities(&self) -> impl Iterator<Item = usize> + '_ {
        self.mandatory_dense_layouts
            .iter()
            .filter_map(|layout| match layout {
                DenseGroupLayout::FullWidth { num_vars } => Some(*num_vars),
                DenseGroupLayout::Bounded { .. } => None,
            })
    }

    pub(crate) fn final_group(&self) -> PolynomialGroupLayout {
        self.final_group
    }

    /// Provision and audit the exact grouped rows before constructing backend matrices.
    /// Used by preprocessing and the offline grouped-schedule diagnostic.
    pub fn extend_catalog(
        &self,
        dense_catalog: &ValidatedScheduleCatalog,
        full_dense_catalog: &ValidatedScheduleCatalog,
        one_hot_catalog: &ValidatedScheduleCatalog,
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        with_one_hot_family!(family, |Cfg| {
            let rows = provision_groups_for_config::<Cfg>(
                dense_catalog,
                full_dense_catalog,
                one_hot_catalog,
                self,
                family,
            )?;
            extend_catalog::<Cfg>(one_hot_catalog, &rows)
        })
    }
}

/// Rows adapted for one concrete setup before they are frozen into a catalog.
#[derive(Clone, Debug, Default)]
pub struct RegisteredRows {
    by_digest: HashMap<ScheduleRowDigest, ResolvedScheduleRow>,
}

impl RegisteredRows {
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &ResolvedScheduleRow> {
        self.by_digest.values()
    }

    fn insert(&mut self, row: ResolvedScheduleRow) -> Result<(), AkitaError> {
        match self.by_digest.entry(row.selection().row_digest) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(row);
                Ok(())
            }
            Entry::Occupied(_) => Err(AkitaError::InvalidSetup(
                "duplicate schedule row digest in provisioned rows".to_owned(),
            )),
        }
    }
}

/// Freeze base and setup-specific rows into one validated immutable catalog.
pub fn extend_catalog<Cfg: CommitmentConfig>(
    base: &ValidatedScheduleCatalog,
    extra: &RegisteredRows,
) -> Result<ValidatedScheduleCatalog, AkitaError> {
    akita_config::validate_config_policy::<Cfg>()?;
    base.validate_binding(
        Cfg::schedule_family_name(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    let rows = base
        .rows()
        .chain(extra.rows())
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .collect::<Vec<_>>();
    ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
}

fn plan_row<Cfg: CommitmentConfig>(
    base: &ValidatedScheduleCatalog,
    final_group: PolynomialGroupLayout,
    producers: &[PrecommittedProducer],
) -> Result<Option<ResolvedScheduleRow>, AkitaError> {
    let request = GroupedGenerationRequest::new(final_group, producers.to_vec());
    let key = request.key();
    if base.resolve_key(&key).is_ok() {
        return Ok(None);
    }
    let main_row = base.resolve_key(&ScheduleLookupKey::single(key.final_group))?;
    let full_width_producers = producers
        .iter()
        .filter(|producer| {
            !producer
                .source_contract()
                .decomposition()
                .has_bounded_committed_source()
        })
        .count();
    if full_width_producers > 1 || (full_width_producers == 1 && producers.len() > 3) {
        return Err(AkitaError::UnsupportedSchedule(
            "full-width batches support one field increment and at most two advice groups"
                .to_owned(),
        ));
    }
    let schedule = find_adapted_schedule(
        main_row,
        &request,
        Cfg::committed_source_contract()?,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?
    .schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds,
    };
    ResolvedScheduleRow::try_new(profiles, schedule, &policy_of::<Cfg>()).map(Some)
}

fn provision_producers<Cfg: CommitmentConfig>(
    base: &ValidatedScheduleCatalog,
    group_combinations: &[Vec<PrecommittedProducer>],
    final_group: PolynomialGroupLayout,
) -> Result<RegisteredRows, AkitaError> {
    akita_config::validate_config_policy::<Cfg>()?;
    base.validate_binding(
        Cfg::schedule_family_name(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    if group_combinations.iter().any(Vec::is_empty) {
        return Err(AkitaError::InvalidSetup(
            "a grouped row must have at least one auxiliary group".to_owned(),
        ));
    }
    if group_combinations.len() > MAX_PROVISIONED_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "provisioning {} rows exceeds the {MAX_PROVISIONED_ROWS}-row cap",
            group_combinations.len()
        )));
    }

    let workers = akita_planner::emit::offline_planning_worker_count(group_combinations.len());
    let planned = akita_planner::emit::bounded_parallel_filter_map(
        group_combinations,
        workers,
        |producers| {
            plan_row::<Cfg>(base, final_group, producers).map_err(|error| error.to_string())
        },
    )
    .map_err(AkitaError::InvalidSetup)?;

    let mut rows = RegisteredRows::default();
    for row in planned {
        rows.insert(row)?;
    }
    Ok(rows)
}

/// Resolve the frozen profile of an independently committed dense object.
pub fn dense_group_profile(
    dense_catalog: &ValidatedScheduleCatalog,
    layout: PolynomialGroupLayout,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    Ok(dense_catalog
        .resolve_key(&ScheduleLookupKey::single(layout))?
        .profiles()
        .final_group)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AdvicePrecommitLayouts {
    untrusted: Option<PolynomialGroupLayout>,
    trusted: Option<PolynomialGroupLayout>,
}

impl AdvicePrecommitLayouts {
    fn precommit_combinations(
        self,
        dense_catalog: &ValidatedScheduleCatalog,
    ) -> Result<Vec<Vec<GroupCommitPhaseParams>>, AkitaError> {
        let untrusted = self
            .untrusted
            .map(|layout| dense_group_profile(dense_catalog, layout))
            .transpose()?;
        let trusted = self
            .trusted
            .map(|layout| dense_group_profile(dense_catalog, layout))
            .transpose()?;
        let mut combinations = Vec::with_capacity(3);
        let mut push_unique = |combination: Vec<GroupCommitPhaseParams>| {
            if !combinations.contains(&combination) {
                combinations.push(combination);
            }
        };
        if let Some(untrusted) = untrusted {
            push_unique(vec![untrusted]);
        }
        if let Some(trusted) = trusted {
            push_unique(vec![trusted]);
        }
        if let (Some(untrusted), Some(trusted)) = (untrusted, trusted) {
            push_unique(vec![untrusted, trusted]);
        }
        Ok(combinations)
    }
}

pub const FIXTURE_TRUSTED_ADVICE_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(14, 1);
pub const FIXTURE_K16_FINAL_NUM_VARS: (usize, usize) = (22, 26);

fn provision_groups_for_config<Cfg: CommitmentConfig>(
    dense_catalog: &ValidatedScheduleCatalog,
    full_dense_catalog: &ValidatedScheduleCatalog,
    one_hot_catalog: &ValidatedScheduleCatalog,
    params: &GroupedScheduleParams,
    family: OneHotFamily,
) -> Result<RegisteredRows, AkitaError> {
    let final_group = params.final_group;
    family.validate_num_vars(final_group.num_vars(), final_group.num_polynomials())?;
    akita_config::validate_config_policy::<JoltDenseBounded>()?;
    dense_catalog.validate_binding(
        JoltDenseBounded::schedule_family_name(),
        &policy_of::<JoltDenseBounded>(),
        JoltDenseBounded::ring_challenge_config,
    )?;
    full_dense_catalog.validate_binding(
        JoltDenseFull::schedule_family_name(),
        &policy_of::<JoltDenseFull>(),
        JoltDenseFull::ring_challenge_config,
    )?;
    let layouts = AdvicePrecommitLayouts {
        untrusted: params
            .untrusted_physical_arity
            .map(|vars| PolynomialGroupLayout::new(vars, 1)),
        trusted: params
            .trusted_physical_arity
            .map(|vars| PolynomialGroupLayout::new(vars, 1)),
    };
    let mandatory = params
        .mandatory_dense_layouts
        .iter()
        .map(|layout| layout.producer(dense_catalog, full_dense_catalog))
        .collect::<Result<Vec<_>, _>>()?;
    let mut combinations = layouts
        .precommit_combinations(dense_catalog)?
        .into_iter()
        .map(|profiles| profiles.iter().map(producer::<JoltDenseBounded>).collect())
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    if mandatory.is_empty() {
        if combinations.is_empty() {
            return Ok(RegisteredRows::default());
        }
    } else {
        for combination in &mut combinations {
            combination.extend(mandatory.iter().copied());
        }
        if !combinations.contains(&mandatory) {
            combinations.push(mandatory);
        }
    }
    let main_row = one_hot_catalog
        .resolve_key(&ScheduleLookupKey::single(final_group))
        .map_err(|error| {
            let guidance = if family.k() == AKITA_ONE_HOT_K256 {
                "; shipped K=256 trace catalogs cover fixture and benchmark shapes only; use K=16 for production traces or supply a catalog containing the exact K=256 shape"
            } else {
                ""
            };
            AkitaError::InvalidSetup(format!(
                "one-hot K={} final shape (num_vars={}, num_polys={}) is outside the admitted catalog{guidance}: {error}",
                family.k(), final_group.num_vars(), final_group.num_polynomials(),
            ))
        })?;
    if family.profile() == AkitaChunkProfile::Single
        && params.full_width_arities().next().is_none()
        && main_row.schedule().recursive_folds.is_empty()
    {
        // Bounded-only Single admission follows the scalar guide's child-fold
        // capability; full planner search is intentionally not attempted here.
        return Err(AkitaError::UnsupportedSchedule(format!(
            "one-hot K={} profile {:?} final arity {} has no recursive child fold in its scalar guide; bounded grouped provisioning requires one; requested groups: {params:?}",
            family.k(), family.profile(), final_group.num_vars()
        )));
    }
    provision_producers::<Cfg>(one_hot_catalog, &combinations, final_group)
}

/// Adapt grouped rows for the standard single-chunk one-hot family.
pub fn provision_groups_for_k(
    dense_catalog: &ValidatedScheduleCatalog,
    full_dense_catalog: &ValidatedScheduleCatalog,
    one_hot_catalog: &ValidatedScheduleCatalog,
    params: &GroupedScheduleParams,
    one_hot_k: usize,
) -> Result<RegisteredRows, AkitaError> {
    let family = OneHotFamily::from_parts(one_hot_k, AkitaChunkProfile::Single)?;
    with_one_hot_family!(family, |Cfg| {
        provision_groups_for_config::<Cfg>(
            dense_catalog,
            full_dense_catalog,
            one_hot_catalog,
            params,
            family,
        )
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "schedule tests should fail loudly")]
mod tests {
    use super::*;
    use crate::adapters::AkitaScheduleArtifacts;
    use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

    #[test]
    fn single_grouped_advice_rejects_small_scalar_arities() {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let dense = artifacts.dense_catalog().unwrap();
        let full_dense = artifacts.full_dense_catalog().unwrap();
        for (k, last_scalar_arity) in [(AKITA_ONE_HOT_K16, 15), (AKITA_ONE_HOT_K256, 16)] {
            let base = artifacts.one_hot_catalog(k).unwrap();
            for final_arity in 12..=last_scalar_arity {
                let params = GroupedScheduleParams::new(
                    None,
                    Some(14),
                    Vec::new(),
                    PolynomialGroupLayout::new(final_arity, 1),
                );
                let error = params
                    .extend_catalog(&dense, &full_dense, &base, k, AkitaChunkProfile::Single)
                    .unwrap_err();
                assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
                let message = error.to_string();
                assert!(
                    message.contains(&format!("K={k} profile Single final arity {final_arity}"))
                );
                assert!(message.contains("trusted_physical_arity: Some(14)"));
            }
        }
    }

    #[test]
    fn grouped_advice_rows_cover_recursive_cutover() {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let dense = artifacts.dense_catalog().unwrap();
        let full_dense = artifacts.full_dense_catalog().unwrap();
        for profile in [
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            let base = artifacts
                .one_hot_catalog_for_profile(AKITA_ONE_HOT_K16, profile)
                .unwrap();
            for (final_num_vars, num_polys) in [(31, 1), (32, 1), (25, 51), (26, 51)] {
                for untrusted in [21, 22] {
                    for trusted in [21, 22] {
                        let params = GroupedScheduleParams::new(
                            Some(untrusted),
                            Some(trusted),
                            Vec::new(),
                            PolynomialGroupLayout::new(final_num_vars, num_polys),
                        );
                        let catalog = params
                            .extend_catalog(&dense, &full_dense, &base, AKITA_ONE_HOT_K16, profile)
                            .unwrap();
                        let key = ScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_num_vars, num_polys),
                            precommitteds: [untrusted, trusted]
                                .into_iter()
                                .map(|num_vars| {
                                    dense_group_profile(
                                        &dense,
                                        PolynomialGroupLayout::new(num_vars, 1),
                                    )
                                    .unwrap()
                                })
                                .collect(),
                        };
                        let row = catalog.resolve_key(&key).unwrap();
                        assert_eq!(row.profiles().precommitteds, key.precommitteds);
                        assert_eq!(
                            row.schedule().root.params.witness_chunk,
                            profile.witness_cfg()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_rows_preserve_chunk_independent_producer_profiles() {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let dense = artifacts.dense_catalog().unwrap();
        let producers = [14, 22].map(|num_vars| {
            dense_group_profile(&dense, PolynomialGroupLayout::new(num_vars, 1)).unwrap()
        });
        let full_dense = artifacts.full_dense_catalog().unwrap();
        let full_producer =
            dense_group_profile(&full_dense, PolynomialGroupLayout::new(14, 1)).unwrap();
        for profile in [
            AkitaChunkProfile::Single,
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            for (one_hot_k, final_num_vars) in [(AKITA_ONE_HOT_K16, 22), (AKITA_ONE_HOT_K256, 20)] {
                let base = artifacts
                    .one_hot_catalog_for_profile(one_hot_k, profile)
                    .unwrap();
                for mandatory in [
                    Vec::new(),
                    vec![DenseGroupLayout::FullWidth { num_vars: 14 }],
                ] {
                    let has_full_producer = !mandatory.is_empty();
                    let params = GroupedScheduleParams::new(
                        Some(14),
                        Some(22),
                        mandatory,
                        PolynomialGroupLayout::new(final_num_vars, 1),
                    );
                    let catalog = params
                        .extend_catalog(&dense, &full_dense, &base, one_hot_k, profile)
                        .unwrap();
                    for mut precommitteds in
                        [vec![producers[0]], vec![producers[1]], producers.to_vec()]
                    {
                        if has_full_producer {
                            precommitteds.push(full_producer);
                        }
                        let key = ScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_num_vars, 1),
                            precommitteds,
                        };
                        let row = catalog.resolve_key(&key).unwrap();
                        assert_eq!(row.profiles().precommitteds, key.precommitteds);
                        assert_eq!(
                            row.schedule().root.params.witness_chunk,
                            profile.witness_cfg()
                        );
                    }
                }
            }
        }
    }
}
