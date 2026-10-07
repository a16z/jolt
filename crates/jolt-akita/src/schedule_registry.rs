//! Setup-owned grouped schedule catalog construction.
//!
//! Base scalar rows come from checked-in external artifacts. Program-specific
//! advice and committed-program shapes, and the field-digit groups committed
//! after the trace, are guided from the approved scalar row during
//! preprocessing and merged into a new immutable catalog owned by that setup.
//! No process-global schedule state participates in proving or verification.

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

use crate::configs::{
    JoltDenseBounded, JoltFieldDigits, JoltOneHotK16, JoltOneHotK256, JoltSignedBytes,
};
use crate::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS, SIGNED_BYTE_NUM_VARS};

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
    #[serde(default)]
    field_digit_groups: Vec<PolynomialGroupLayout>,
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
            field_digit_groups: Vec::new(),
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

    /// `(num_vars, num_polys)` of each field-digit group, in role order after
    /// every program object.
    pub fn with_field_digit_groups(
        mut self,
        groups: impl IntoIterator<Item = (usize, usize)>,
    ) -> Self {
        self.field_digit_groups = groups
            .into_iter()
            .map(|(num_vars, num_polys)| PolynomialGroupLayout::new(num_vars, num_polys))
            .collect();
        self
    }

    pub(crate) fn final_num_vars(&self) -> usize {
        self.final_arity
    }

    pub(crate) fn field_digit_groups(&self) -> &[PolynomialGroupLayout] {
        &self.field_digit_groups
    }

    pub(crate) fn capacity(&self) -> (usize, usize, usize) {
        let single = |num_vars| PolynomialGroupLayout::new(num_vars, 1);
        std::iter::once(single(self.final_arity))
            .chain(
                self.untrusted_physical_arity
                    .into_iter()
                    .chain(self.trusted_physical_arity)
                    .chain(self.direct_program_physical_arities.iter().copied())
                    .map(single),
            )
            .chain(self.field_digit_groups.iter().copied())
            .fold((0, 0, 0), |(num_vars, polys, total), group| {
                (
                    num_vars.max(group.num_vars()),
                    polys.max(group.num_polynomials()),
                    total + group.num_polynomials(),
                )
            })
    }

    /// Adapt grouped rows for optional advice followed by committed-program
    /// objects and then the field-digit groups, all present in every row, and
    /// freeze them with `base` into one setup-owned catalog.
    pub fn extend_catalog(
        &self,
        dense_catalog: &ValidatedScheduleCatalog,
        field_digit_profiles: &[GroupCommitPhaseParams],
        base: &ValidatedScheduleCatalog,
        trace: TraceFamily,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        akita_config::validate_config_policy::<JoltDenseBounded>()?;
        dense_catalog.validate_binding(
            JoltDenseBounded::schedule_family_name(),
            &policy_of::<JoltDenseBounded>(),
            JoltDenseBounded::ring_challenge_config,
        )?;
        let mandatory = self
            .direct_program_physical_arities
            .iter()
            .map(|num_vars| {
                precommitted_producer::<JoltDenseBounded>(dense_precommit_profile(
                    dense_catalog,
                    PolynomialGroupLayout::new(*num_vars, 1),
                )?)
            })
            .chain(
                field_digit_profiles
                    .iter()
                    .map(|profile| precommitted_producer::<JoltFieldDigits>(*profile)),
            )
            .collect::<Result<Vec<_>, _>>()?;
        let layouts = AdvicePrecommitLayouts {
            untrusted: self
                .untrusted_physical_arity
                .map(|num_vars| PolynomialGroupLayout::new(num_vars, 1)),
            trusted: self
                .trusted_physical_arity
                .map(|num_vars| PolynomialGroupLayout::new(num_vars, 1)),
        };
        let mut combinations = layouts.precommit_combinations(dense_catalog)?;
        if !mandatory.is_empty() {
            for combination in &mut combinations {
                combination.extend(mandatory.iter().copied());
            }
            if !combinations.contains(&mandatory) {
                combinations.push(mandatory);
            }
        }
        let final_num_vars = self.final_arity;
        let (min, max) = match trace {
            TraceFamily::OneHotK16 => K16_NUM_VARS,
            TraceFamily::OneHotK256 => K256_NUM_VARS,
            TraceFamily::SignedBytes => SIGNED_BYTE_NUM_VARS,
        };
        if !combinations.is_empty() && !(min..=max).contains(&final_num_vars) {
            return Err(AkitaError::InvalidSetup(format!(
                "{trace:?} final arity {final_num_vars} is outside the supported range {min}..={max}"
            )));
        }
        match trace {
            TraceFamily::OneHotK16 => extend_catalog::<JoltOneHotK16>(
                base,
                &provision::<JoltOneHotK16>(base, &combinations, [final_num_vars])?,
            ),
            TraceFamily::OneHotK256 => extend_catalog::<JoltOneHotK256>(
                base,
                &provision::<JoltOneHotK256>(base, &combinations, [final_num_vars])?,
            ),
            TraceFamily::SignedBytes => extend_catalog::<JoltSignedBytes>(
                base,
                &provision::<JoltSignedBytes>(base, &combinations, [final_num_vars])?,
            ),
        }
    }
}

/// Schedule family of a setup's final trace group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceFamily {
    OneHotK16,
    OneHotK256,
    SignedBytes,
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
    request: &GroupedGenerationRequest,
) -> Result<ResolvedScheduleRow, AkitaError> {
    let key = request.key();
    let main_row = base.resolve_key(&AkitaScheduleLookupKey::single(key.final_group))?;
    let planned = find_adapted_schedule(
        main_row,
        request,
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
        precommitteds: key.precommitteds,
    };
    ResolvedScheduleRow::try_new(profiles, schedule, &policy_of::<Cfg>())
}

/// Binds a frozen precommitted profile to the contract of the family that
/// commits it.
pub fn precommitted_producer<Cfg: CommitmentConfig>(
    profile: GroupCommitPhaseParams,
) -> Result<PrecommittedProducer, AkitaError> {
    PrecommittedProducer::try_new(
        profile,
        Cfg::committed_source_contract()?,
        honest_fold_policy_of::<Cfg>(),
    )
}

/// Adapt missing grouped rows from the base catalog's approved scalar rows.
pub fn provision<Cfg: CommitmentConfig>(
    base: &ValidatedScheduleCatalog,
    precommitted_combinations: &[Vec<PrecommittedProducer>],
    final_num_vars: impl IntoIterator<Item = usize>,
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
    let final_arities = final_num_vars.into_iter().collect::<Vec<_>>();
    let requests = precommitted_combinations
        .iter()
        .flat_map(|producers| {
            final_arities.iter().map(|num_vars| {
                GroupedGenerationRequest::new(
                    PolynomialGroupLayout::new(*num_vars, 1),
                    producers.clone(),
                )
            })
        })
        .collect::<Vec<_>>();
    if requests.len() > MAX_PROVISIONED_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "provisioning {} rows exceeds the {MAX_PROVISIONED_ROWS}-row cap",
            requests.len()
        )));
    }

    let workers = akita_planner::emit::offline_planning_worker_count(requests.len());
    let planned = akita_planner::emit::bounded_parallel_filter_map(&requests, workers, |request| {
        if base.resolve_key(&request.key()).is_ok() {
            return Ok(None);
        }
        plan_row::<Cfg>(base, request)
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
    ) -> Result<Vec<Vec<PrecommittedProducer>>, AkitaError> {
        let dense_producer = |layout| {
            precommitted_producer::<JoltDenseBounded>(dense_precommit_profile(
                dense_catalog,
                layout,
            )?)
        };
        let untrusted = self.untrusted.map(dense_producer).transpose()?;
        let trusted = self.trusted.map(dense_producer).transpose()?;
        let mut combinations = Vec::with_capacity(3);
        let mut push_unique = |combination: Vec<PrecommittedProducer>| {
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
