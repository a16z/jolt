//! Setup-owned grouped schedule catalog construction.
//!
//! Base scalar rows come from checked-in external artifacts. Program-specific
//! advice and committed-program shapes are guided from the approved scalar row
//! during preprocessing and merged into a new immutable catalog owned by that
//! setup. No process-global schedule state participates in proving or
//! verification.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
use akita_planner::find_adapted_schedule;
use akita_schedules::{ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    PolynomialGroupLayout, ScheduleRowDigest,
};
use serde::{Deserialize, Serialize};

use crate::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
use crate::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS};
use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

/// Upper bound on rows planned by one preprocessing request.
const MAX_PROVISIONED_ROWS: usize = 128;

/// Public inputs needed to construct this setup's grouped schedules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecommittedScheduleParams {
    untrusted_physical_arity: Option<usize>,
    trusted_physical_arity: Option<usize>,
    #[serde(default)]
    direct_program_physical_arities: Vec<usize>,
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
            direct_program_physical_arities: Vec::new(),
            final_arity: final_num_vars,
        }
    }

    pub fn with_direct_program_physical_arities(
        mut self,
        direct_program_physical_arities: Vec<usize>,
    ) -> Self {
        self.direct_program_physical_arities = direct_program_physical_arities;
        self
    }

    pub(crate) fn final_num_vars(&self) -> usize {
        self.final_arity
    }

    pub(crate) fn extend_catalog(
        &self,
        dense_catalog: &TrustedScheduleCatalog,
        one_hot_catalog: &TrustedScheduleCatalog,
        one_hot_k: usize,
    ) -> Result<TrustedScheduleCatalog, AkitaError> {
        let rows = provision_precommitted_for_k(
            dense_catalog,
            one_hot_catalog,
            self.untrusted_physical_arity,
            self.trusted_physical_arity,
            &self.direct_program_physical_arities,
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
    base: &TrustedScheduleCatalog,
    extra: &RegisteredRows,
) -> Result<TrustedScheduleCatalog, AkitaError> {
    akita_config::validate_trusted_schedule_catalog::<Cfg>(base)?;
    let rows = base
        .rows()
        .chain(extra.rows())
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .collect::<Vec<_>>();
    TrustedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
}

fn plan_row<Cfg: CommitmentConfig, ProducerCfg: CommitmentConfig>(
    base: &TrustedScheduleCatalog,
    key: &AkitaScheduleLookupKey,
) -> Result<ResolvedScheduleRow, AkitaError> {
    let main_row = base.resolve_key(&AkitaScheduleLookupKey::single(key.final_group))?;
    let producer_contract = ProducerCfg::committed_source_contract()?;
    let producer_fold_policy = honest_fold_policy_of::<ProducerCfg>();
    let producers = key
        .precommitteds
        .iter()
        .copied()
        .map(|profile| {
            PrecommittedProducer::try_new(profile, producer_contract, producer_fold_policy)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let request = GroupedGenerationRequest::new(key.final_group, producers);
    let planned = find_adapted_schedule(
        &main_row,
        &request,
        honest_fold_policy_of::<Cfg>(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    let schedule = planned.schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    };
    ResolvedScheduleRow::try_new(profiles, schedule, &policy_of::<Cfg>())
}

/// Adapt missing grouped rows from the base catalog's approved scalar rows.
pub fn provision<Cfg: CommitmentConfig, ProducerCfg: CommitmentConfig>(
    base: &TrustedScheduleCatalog,
    precommitted_combinations: &[Vec<GroupCommitPhaseParams>],
    final_num_vars: impl IntoIterator<Item = usize>,
) -> Result<RegisteredRows, AkitaError> {
    akita_config::validate_trusted_schedule_catalog::<Cfg>(base)?;
    if precommitted_combinations.iter().any(Vec::is_empty) {
        return Err(AkitaError::InvalidSetup(
            "a grouped row must have at least one precommitted group".to_owned(),
        ));
    }
    let final_arities = final_num_vars.into_iter().collect::<Vec<_>>();
    let keys = precommitted_combinations
        .iter()
        .flat_map(|precommitteds| {
            final_arities.iter().map(|num_vars| AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(*num_vars, 1),
                precommitteds: precommitteds.clone(),
            })
        })
        .collect::<Vec<_>>();
    if keys.len() > MAX_PROVISIONED_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "provisioning {} rows exceeds the {MAX_PROVISIONED_ROWS}-row cap",
            keys.len()
        )));
    }

    let workers = akita_planner::emit::offline_planning_worker_count(keys.len());
    let planned = akita_planner::emit::bounded_parallel_filter_map(&keys, workers, |key| {
        if base.resolve_key(key).is_ok() {
            return Ok(None);
        }
        plan_row::<Cfg, ProducerCfg>(base, key)
            .map(Some)
            .map_err(|error| error.to_string())
    })
    .map_err(AkitaError::InvalidSetup)?;

    let mut rows = RegisteredRows::default();
    for row in planned {
        rows.insert(row)?;
    }
    Ok(rows)
}

/// Resolve the frozen profile of an independently committed dense object.
pub fn dense_precommit_profile(
    dense_catalog: &TrustedScheduleCatalog,
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
        dense_catalog: &TrustedScheduleCatalog,
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

/// Adapt grouped rows for optional advice followed by committed-program objects.
pub fn provision_precommitted_for_k(
    dense_catalog: &TrustedScheduleCatalog,
    one_hot_catalog: &TrustedScheduleCatalog,
    untrusted_physical_vars: Option<usize>,
    trusted_physical_vars: Option<usize>,
    direct_program_physical_vars: &[usize],
    one_hot_k: usize,
    final_num_vars: usize,
) -> Result<RegisteredRows, AkitaError> {
    akita_config::validate_trusted_schedule_catalog::<JoltDenseBounded>(dense_catalog)?;
    let layouts = AdvicePrecommitLayouts {
        untrusted: untrusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
        trusted: trusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
    };
    let mandatory = direct_program_physical_vars
        .iter()
        .map(|vars| dense_precommit_profile(dense_catalog, PolynomialGroupLayout::new(*vars, 1)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut combinations = layouts.precommit_combinations(dense_catalog)?;
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
        AKITA_ONE_HOT_K256 => provision::<JoltOneHotK256, JoltDenseBounded>(
            one_hot_catalog,
            &combinations,
            [final_num_vars],
        ),
        AKITA_ONE_HOT_K16 => provision::<JoltOneHotK16, JoltDenseBounded>(
            one_hot_catalog,
            &combinations,
            [final_num_vars],
        ),
        _ => unreachable!("one-hot K was validated above"),
    }
}
