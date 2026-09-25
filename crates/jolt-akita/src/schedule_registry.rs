//! Setup-owned grouped schedule catalog construction.
//!
//! Base scalar rows come from checked-in external artifacts. Program-specific
//! advice and committed-program shapes are guided from the approved scalar row
//! during preprocessing. A field increment with optional advice may require a
//! new grouped schedule when that scalar row's fixed recursion geometry is
//! infeasible. Every planned row is audited and merged into the setup's immutable
//! catalog; runtime proving and verification never plan schedules.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
use akita_planner::find_adapted_schedule;
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    PolynomialGroupLayout, ScheduleRowDigest,
};
use serde::{Deserialize, Serialize};

use crate::configs::{JoltDenseBounded, JoltDenseFull, JoltOneHotK16, JoltOneHotK256};
use crate::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS};
use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

/// Upper bound on rows planned by one preprocessing request.
const MAX_PROVISIONED_ROWS: usize = 128;

/// Physical shape and admitted coefficient range of one dense prefix group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DensePrecommitLayout {
    Bounded { num_vars: usize },
    FullWidth { num_vars: usize },
}

impl DensePrecommitLayout {
    fn producer(
        self,
        bounded: &ValidatedScheduleCatalog,
        full_width: &ValidatedScheduleCatalog,
    ) -> Result<PrecommittedProducer, AkitaError> {
        match self {
            Self::Bounded { num_vars } => producer::<JoltDenseBounded>(&dense_precommit_profile(
                bounded,
                PolynomialGroupLayout::new(num_vars, 1),
            )?),
            Self::FullWidth { num_vars } => producer::<JoltDenseFull>(&dense_precommit_profile(
                full_width,
                PolynomialGroupLayout::new(num_vars, 1),
            )?),
        }
    }
}

fn producer<Cfg: CommitmentConfig>(
    profile: &GroupCommitPhaseParams,
) -> Result<PrecommittedProducer, AkitaError> {
    PrecommittedProducer::try_new(
        *profile,
        Cfg::committed_source_contract()?,
        honest_fold_policy_of::<Cfg>(),
    )
}

/// Public inputs needed to construct this setup's grouped schedules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecommittedScheduleParams {
    untrusted_physical_arity: Option<usize>,
    trusted_physical_arity: Option<usize>,
    #[serde(default)]
    mandatory_dense_layouts: Vec<DensePrecommitLayout>,
    final_arity: usize,
}

impl PrecommittedScheduleParams {
    pub fn new(
        untrusted_physical_num_vars: Option<usize>,
        trusted_physical_num_vars: Option<usize>,
        final_num_vars: usize,
    ) -> Self {
        Self {
            untrusted_physical_arity: untrusted_physical_num_vars,
            trusted_physical_arity: trusted_physical_num_vars,
            mandatory_dense_layouts: Vec::new(),
            final_arity: final_num_vars,
        }
    }

    pub fn with_mandatory_dense_layouts(
        mut self,
        mandatory_dense_layouts: Vec<DensePrecommitLayout>,
    ) -> Self {
        self.mandatory_dense_layouts = mandatory_dense_layouts;
        self
    }

    pub(crate) fn final_num_vars(&self) -> usize {
        self.final_arity
    }

    pub(crate) fn extend_catalog(
        &self,
        dense_catalog: &ValidatedScheduleCatalog,
        full_dense_catalog: &ValidatedScheduleCatalog,
        one_hot_catalog: &ValidatedScheduleCatalog,
        one_hot_k: usize,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        let rows = provision_precommitted_for_k(
            dense_catalog,
            full_dense_catalog,
            one_hot_catalog,
            self.untrusted_physical_arity,
            self.trusted_physical_arity,
            &self.mandatory_dense_layouts,
            one_hot_k,
            self.final_arity,
        )?;
        match one_hot_k {
            AKITA_ONE_HOT_K16 => extend_catalog::<JoltOneHotK16>(one_hot_catalog, &rows),
            AKITA_ONE_HOT_K256 => extend_catalog::<JoltOneHotK256>(one_hot_catalog, &rows),
            other => Err(AkitaError::InvalidSetup(format!(
                "unsupported one-hot K {other} for grouped schedule catalog"
            ))),
        }
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
    final_num_vars: usize,
    producers: &[PrecommittedProducer],
) -> Result<Option<ResolvedScheduleRow>, AkitaError> {
    let request = GroupedGenerationRequest::new(
        PolynomialGroupLayout::new(final_num_vars, 1),
        producers.to_vec(),
    );
    let key = request.key();
    if base.resolve_key(&key).is_ok() {
        return Ok(None);
    }
    let main_row = base.resolve_key(&AkitaScheduleLookupKey::single(key.final_group))?;
    let adapted = find_adapted_schedule(
        main_row,
        &request,
        honest_fold_policy_of::<Cfg>(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    );
    let schedule = match adapted {
        Ok(planned) => planned.schedule,
        Err(AkitaError::UnsupportedSchedule(_))
            if producers.len() <= 3
                && producers
                    .iter()
                    .filter(|producer| {
                        !producer
                            .source_contract()
                            .decomposition()
                            .has_bounded_committed_source()
                    })
                    .count()
                    == 1 =>
        {
            // FieldRdInc plus at most two advice groups is the only supported
            // full-width batch. Restrict full search to that shape so it cannot
            // bypass the adapted planner's opening-assignment budget for larger
            // batches. Every prefix commitment's descriptor remains fixed.
            let fold_policies = producers
                .iter()
                .map(|producer| {
                    let contract = producer.source_contract();
                    contract
                        .class()
                        .honest_fold_policy(contract.decomposition().field_bits())
                })
                .collect::<Vec<_>>();
            crate::planning::plan_schedule::<Cfg>(&key, &fold_policies)?
        }
        Err(error) => return Err(error),
    };
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
    precommitted_combinations: &[Vec<PrecommittedProducer>],
    final_num_vars: usize,
) -> Result<RegisteredRows, AkitaError> {
    akita_config::validate_config_policy::<Cfg>()?;
    base.validate_binding(
        Cfg::schedule_family_name(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    if precommitted_combinations.iter().any(Vec::is_empty) {
        return Err(AkitaError::InvalidSetup(
            "a grouped row must have at least one precommitted group".to_owned(),
        ));
    }
    if precommitted_combinations.len() > MAX_PROVISIONED_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "provisioning {} rows exceeds the {MAX_PROVISIONED_ROWS}-row cap",
            precommitted_combinations.len()
        )));
    }

    let workers =
        akita_planner::emit::offline_planning_worker_count(precommitted_combinations.len());
    let planned = akita_planner::emit::bounded_parallel_filter_map(
        precommitted_combinations,
        workers,
        |producers| {
            plan_row::<Cfg>(base, final_num_vars, producers).map_err(|error| error.to_string())
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
pub fn dense_precommit_profile(
    dense_catalog: &ValidatedScheduleCatalog,
    layout: PolynomialGroupLayout,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    Ok(dense_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(layout))?
        .profiles()
        .final_group)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdvicePrecommitLayouts {
    pub untrusted: Option<PolynomialGroupLayout>,
    pub trusted: Option<PolynomialGroupLayout>,
}

impl AdvicePrecommitLayouts {
    fn precommit_combinations(
        self,
        dense_catalog: &ValidatedScheduleCatalog,
    ) -> Result<Vec<Vec<GroupCommitPhaseParams>>, AkitaError> {
        let untrusted = self
            .untrusted
            .map(|layout| dense_precommit_profile(dense_catalog, layout))
            .transpose()?;
        let trusted = self
            .trusted
            .map(|layout| dense_precommit_profile(dense_catalog, layout))
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

/// Adapt grouped rows for optional advice followed by mandatory dense objects,
/// all in canonical precommit order.
#[expect(
    clippy::too_many_arguments,
    reason = "grouped provisioning combines two dense producer catalogs with trace and object shapes"
)]
pub fn provision_precommitted_for_k(
    dense_catalog: &ValidatedScheduleCatalog,
    full_dense_catalog: &ValidatedScheduleCatalog,
    one_hot_catalog: &ValidatedScheduleCatalog,
    untrusted_physical_vars: Option<usize>,
    trusted_physical_vars: Option<usize>,
    mandatory_dense_layouts: &[DensePrecommitLayout],
    one_hot_k: usize,
    final_num_vars: usize,
) -> Result<RegisteredRows, AkitaError> {
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
        untrusted: untrusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
        trusted: trusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
    };
    let mandatory = mandatory_dense_layouts
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
    let (min, max) = match one_hot_k {
        AKITA_ONE_HOT_K256 => K256_NUM_VARS,
        AKITA_ONE_HOT_K16 => K16_NUM_VARS,
        other => {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported one-hot K {other} for grouped schedule provisioning"
            )))
        }
    };
    if !(min..=max).contains(&final_num_vars) {
        return Err(AkitaError::InvalidSetup(format!(
            "one-hot K={one_hot_k} final arity {final_num_vars} is outside the supported range {min}..={max}"
        )));
    }
    match one_hot_k {
        AKITA_ONE_HOT_K256 => {
            provision_producers::<JoltOneHotK256>(one_hot_catalog, &combinations, final_num_vars)
        }
        AKITA_ONE_HOT_K16 => {
            provision_producers::<JoltOneHotK16>(one_hot_catalog, &combinations, final_num_vars)
        }
        _ => unreachable!("one-hot K was validated above"),
    }
}
