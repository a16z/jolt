//! Preprocessing-provisioned grouped schedule rows.
//!
//! Advice and direct-program layouts are known during preprocessing, where
//! [`provision`] plans their grouped rows for the static
//! [`CommitmentConfig`](akita_config::CommitmentConfig) resolution hooks.
//! Resolution never invokes the planner.

use std::any::TypeId;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig, ResolvedScheduleRow};
use akita_pcs::AkitaError;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule,
    GroupCommitPhaseParams, OpeningScheduleSelection, PolynomialGroupLayout, ScheduleRowDigest,
};
use serde::{Deserialize, Serialize};

use crate::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
use crate::planning::plan_schedule;
use crate::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS};
use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

/// Upper bound on rows planned by one preprocessing request. Installed rows
/// are immutable and shared as a process-wide cache across independent setups.
pub const MAX_PROVISIONED_ROWS: usize = 128;

/// Public inputs needed to restore this setup's grouped precommitted schedules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecommittedScheduleParams {
    untrusted_physical_arity: Option<usize>,
    trusted_physical_arity: Option<usize>,
    #[serde(default)]
    direct_program_physical_arities: Vec<usize>,
    final_arity: usize,
    /// The always-present FR limb group's arity line (field-inline proofs
    /// only). `None` keeps provisioning identical to the base protocol.
    #[cfg(feature = "field-inline")]
    field_inc_limbs: Option<FieldIncLimbScheduleParams>,
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
            #[cfg(feature = "field-inline")]
            field_inc_limbs: None,
        }
    }

    /// Attach the FR limb-group arity line: the provisioned rows then carry
    /// the setup arity's FR profile as a mandatory group after the advice
    /// (an FR-on prover commits the group on every proof, so no FR-absent
    /// row is reachable).
    #[cfg(feature = "field-inline")]
    pub fn with_field_inc_limbs(mut self, field_inc_limbs: FieldIncLimbScheduleParams) -> Self {
        self.field_inc_limbs = Some(field_inc_limbs);
        self
    }

    pub fn with_direct_program_physical_arities(
        mut self,
        direct_program_physical_arities: Vec<usize>,
    ) -> Self {
        self.direct_program_physical_arities = direct_program_physical_arities;
        self
    }

    pub(crate) fn provision(&self, one_hot_k: usize) -> Result<RegisteredRows, AkitaError> {
        provision_precommitted_for_k(
            self.untrusted_physical_arity,
            self.trusted_physical_arity,
            &self.direct_program_physical_arities,
            #[cfg(feature = "field-inline")]
            self.field_inc_limbs,
            one_hot_k,
            self.final_arity,
        )
    }
}

/// The FR limb group's physical-arity line: `physical = max(log_T +
/// selector_num_vars, min_physical_arity)` with `log_T = final_num_vars -
/// trace_arity_overhead`. All three terms are caller-derived from the
/// jolt-claims packing laws and carried here as serialized data (this crate
/// is claims-free); the FR provisioning pin test holds the line to
/// `FieldIncLimbPackingPlan` across every final arity.
#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldIncLimbScheduleParams {
    trace_arity_overhead: usize,
    min_physical_arity: usize,
    selector_num_vars: usize,
}

#[cfg(feature = "field-inline")]
impl FieldIncLimbScheduleParams {
    pub fn new(
        trace_arity_overhead: usize,
        min_physical_arity: usize,
        selector_num_vars: usize,
    ) -> Self {
        Self {
            trace_arity_overhead,
            min_physical_arity,
            selector_num_vars,
        }
    }

    /// The FR limb group's physical arity at final arity `final_num_vars`,
    /// or `None` below the packed trace's own overhead (no trace exists
    /// there, so no FR pairing either).
    pub fn physical_num_vars(self, final_num_vars: usize) -> Option<usize> {
        let log_t = final_num_vars.checked_sub(self.trace_arity_overhead)?;
        Some(
            log_t
                .checked_add(self.selector_num_vars)?
                .max(self.min_physical_arity),
        )
    }
}

/// One config's preprocessing-provisioned rows.
#[derive(Clone, Debug, Default)]
pub struct RegisteredRows {
    by_digest: HashMap<ScheduleRowDigest, ResolvedScheduleRow>,
}

impl RegisteredRows {
    /// The installed rows, in unspecified order.
    pub fn rows(&self) -> impl Iterator<Item = &ResolvedScheduleRow> {
        self.by_digest.values()
    }

    fn insert(&mut self, row: ResolvedScheduleRow) -> Result<(), AkitaError> {
        let digest = row.selection().row_digest;
        match self.by_digest.entry(digest) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(row);
                Ok(())
            }
            Entry::Occupied(_) => Err(AkitaError::InvalidSetup(
                "duplicate schedule row digest in provisioned rows".to_owned(),
            )),
        }
    }

    fn by_selection(&self, selection: OpeningScheduleSelection) -> Option<&ResolvedScheduleRow> {
        self.by_digest.get(&selection.row_digest)
    }

    fn by_key(&self, key: &AkitaScheduleLookupKey) -> Option<&ResolvedScheduleRow> {
        self.by_digest.values().find(|row| {
            let profiles = row.profiles();
            profiles.final_group.group == key.final_group
                && profiles.precommitteds == key.precommitteds
        })
    }

    fn by_profiles(&self, profiles: &CommittedGroupBatchProfile) -> Option<&ResolvedScheduleRow> {
        self.by_digest
            .values()
            .find(|row| row.profiles() == profiles)
    }
}

type Registry = RwLock<HashMap<TypeId, RegisteredRows>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn poisoned() -> AkitaError {
    AkitaError::InvalidSetup("schedule registry lock poisoned".to_owned())
}

/// The rows installed for `Cfg`, or an empty set.
pub fn registered_rows<Cfg: CommitmentConfig + 'static>() -> Result<RegisteredRows, AkitaError> {
    let guard = registry().read().map_err(|_| poisoned())?;
    Ok(guard.get(&TypeId::of::<Cfg>()).cloned().unwrap_or_default())
}

/// Resolve one public selection against `Cfg`'s installed rows.
pub fn lookup_selection<Cfg: CommitmentConfig + 'static>(
    selection: OpeningScheduleSelection,
) -> Option<ResolvedScheduleRow> {
    let guard = registry().read().ok()?;
    guard
        .get(&TypeId::of::<Cfg>())?
        .by_selection(selection)
        .cloned()
}

/// Resolve one lookup key against `Cfg`'s installed rows.
pub fn lookup_key<Cfg: CommitmentConfig + 'static>(
    key: &AkitaScheduleLookupKey,
) -> Option<ResolvedScheduleRow> {
    let guard = registry().read().ok()?;
    guard.get(&TypeId::of::<Cfg>())?.by_key(key).cloned()
}

/// Resolve one exact committed-profile batch against `Cfg`'s installed rows.
pub fn lookup_profiles<Cfg: CommitmentConfig + 'static>(
    profiles: &CommittedGroupBatchProfile,
) -> Option<ResolvedScheduleRow> {
    let guard = registry().read().ok()?;
    guard
        .get(&TypeId::of::<Cfg>())?
        .by_profiles(profiles)
        .cloned()
}

/// Resolve `key` without consulting the runtime registry.
pub(crate) fn catalog_only_row<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<ResolvedScheduleRow, AkitaError> {
    akita_schedules::resolve_generated_catalog_row_for_key(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::schedule_catalog(),
    )
}

fn plan_row<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
) -> Result<ResolvedScheduleRow, AkitaError> {
    let policy = policy_of::<Cfg>();
    let schedule = plan_schedule::<Cfg>(key, precommitted_honest_fold_policies)?;
    reject_setup_prefix_contributions(&schedule)?;

    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    };
    let selection = OpeningScheduleSelection {
        row_digest: schedule_row_digest(&profiles, &schedule)?,
    };
    ResolvedScheduleRow::try_new(selection, profiles, schedule, &policy)
}

/// Reject schedules that Jolt's verifier-side shape guard cannot admit.
fn reject_setup_prefix_contributions(schedule: &FoldSchedule) -> Result<(), AkitaError> {
    if schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.setup_prefix().is_some())
    {
        return Err(AkitaError::InvalidSetup(
            "provisioned schedule carries a recursive setup-prefix contribution, which Jolt's \
             shape guard does not admit"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Plan and install missing grouped rows across `final_num_vars`.
pub fn provision<Cfg: CommitmentConfig + 'static>(
    precommitted_combinations: &[Vec<GroupCommitPhaseParams>],
    precommitted_honest_fold_policy: HonestFoldPolicySpec,
    final_num_vars: impl IntoIterator<Item = usize>,
) -> Result<RegisteredRows, AkitaError> {
    if precommitted_combinations.iter().any(Vec::is_empty) {
        return Err(AkitaError::InvalidSetup(
            "a grouped row must have at least one precommitted group".to_owned(),
        ));
    }
    let final_arities: Vec<usize> = final_num_vars.into_iter().collect();
    let keys: Vec<AkitaScheduleLookupKey> = precommitted_combinations
        .iter()
        .flat_map(|precommitteds| {
            final_arities
                .iter()
                .map(|num_vars| AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(*num_vars, 1),
                    precommitteds: precommitteds.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if keys.len() > MAX_PROVISIONED_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "provisioning {} rows exceeds the {MAX_PROVISIONED_ROWS}-row cap",
            keys.len()
        )));
    }

    // Planner solves hold large suffix DP caches, so use Akita's bounded worker count.
    let workers = akita_planner::emit::offline_planning_worker_count(keys.len());
    let planned = akita_planner::emit::bounded_parallel_filter_map(&keys, workers, |key| {
        if let Some(row) = lookup_key::<Cfg>(key) {
            return Ok(Some(row));
        }
        if catalog_only_row::<Cfg>(key).is_ok() {
            return Ok(None);
        }
        let policies = vec![precommitted_honest_fold_policy; key.precommitteds.len()];
        match plan_row::<Cfg>(key, &policies) {
            Ok(row) => Ok(Some(row)),
            // A precommit can make the family's smallest final arities unsupported.
            Err(AkitaError::UnsupportedSchedule(reason)) => {
                tracing::debug!(
                    final_num_vars = key.final_group.num_vars(),
                    %reason,
                    "no grouped schedule at this final arity; skipping"
                );
                Ok(None)
            }
            Err(error) => Err(error.to_string()),
        }
    })
    .map_err(AkitaError::InvalidSetup)?;

    let mut rows = RegisteredRows::default();
    for row in planned {
        rows.insert(row)?;
    }
    publish::<Cfg>(rows)
}

/// Publish rows for the static resolution hooks. Identical rows are idempotent.
fn publish<Cfg: CommitmentConfig + 'static>(
    rows: RegisteredRows,
) -> Result<RegisteredRows, AkitaError> {
    let mut guard = registry().write().map_err(|_| poisoned())?;
    let ambient = guard.entry(TypeId::of::<Cfg>()).or_default();
    for (digest, row) in &rows.by_digest {
        if ambient
            .by_digest
            .get(digest)
            .is_some_and(|existing| existing.profiles() != row.profiles())
        {
            return Err(AkitaError::InvalidSetup(
                "schedule row digest collision: same identity, different profiles".to_owned(),
            ));
        }
    }
    for (digest, row) in &rows.by_digest {
        let _ = ambient
            .by_digest
            .entry(*digest)
            .or_insert_with(|| row.clone());
    }
    Ok(rows)
}

/// Resolve the frozen profile produced by an independent dense commit.
pub fn dense_precommit_profile(
    layout: PolynomialGroupLayout,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    JoltDenseBounded::profile_without_precommitted_groups(layout)
}

/// Advice layouts fixed by the program's preprocessing capacities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdvicePrecommitLayouts {
    pub untrusted: Option<PolynomialGroupLayout>,
    pub trusted: Option<PolynomialGroupLayout>,
}

impl AdvicePrecommitLayouts {
    /// Every distinct ordered non-empty advice presence combination.
    /// Equal profiles deduplicate because schedule keys do not encode advice roles.
    fn precommit_combinations(self) -> Result<Vec<Vec<GroupCommitPhaseParams>>, AkitaError> {
        let untrusted = self.untrusted.map(dense_precommit_profile).transpose()?;
        let trusted = self.trusted.map(dense_precommit_profile).transpose()?;
        let mut combinations: Vec<Vec<GroupCommitPhaseParams>> = Vec::with_capacity(3);
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

/// Provision grouped advice rows across the possible final trace arities.
pub fn provision_advice<Cfg: CommitmentConfig + 'static>(
    layouts: AdvicePrecommitLayouts,
    final_num_vars: impl IntoIterator<Item = usize>,
) -> Result<RegisteredRows, AkitaError> {
    let combinations = layouts.precommit_combinations()?;
    if combinations.is_empty() {
        return Ok(RegisteredRows::default());
    }
    provision::<Cfg>(
        &combinations,
        honest_fold_policy_of::<JoltDenseBounded>(),
        final_num_vars,
    )
}

/// Default advice fixture at the dense arity floor.
pub const FIXTURE_TRUSTED_ADVICE_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(14, 1);

/// K=16 fixture final arities for canonical `log_T = 12..=16` traces.
pub const FIXTURE_K16_FINAL_NUM_VARS: (usize, usize) = (22, 26);

/// Provision advice rows before building the packed setup that must cover them.
pub fn provision_advice_for_k(
    untrusted_physical_vars: Option<usize>,
    trusted_physical_vars: Option<usize>,
    one_hot_k: usize,
    final_num_vars: usize,
) -> Result<RegisteredRows, AkitaError> {
    provision_precommitted_for_k(
        untrusted_physical_vars,
        trusted_physical_vars,
        &[],
        #[cfg(feature = "field-inline")]
        None,
        one_hot_k,
        final_num_vars,
    )
}

/// Provision grouped rows for optional advice followed by the mandatory
/// groups — (field-inline) the FR limb group, then the direct
/// committed-program objects — all in canonical precommit order.
pub fn provision_precommitted_for_k(
    untrusted_physical_vars: Option<usize>,
    trusted_physical_vars: Option<usize>,
    direct_program_physical_vars: &[usize],
    #[cfg(feature = "field-inline")] field_inc_limbs: Option<FieldIncLimbScheduleParams>,
    one_hot_k: usize,
    final_num_vars: usize,
) -> Result<RegisteredRows, AkitaError> {
    let layouts = AdvicePrecommitLayouts {
        untrusted: untrusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
        trusted: trusted_physical_vars.map(|vars| PolynomialGroupLayout::new(vars, 1)),
    };
    let mut mandatory = Vec::with_capacity(
        usize::from(cfg!(feature = "field-inline")) + direct_program_physical_vars.len(),
    );
    #[cfg(feature = "field-inline")]
    if let Some(field_inc_limbs) = field_inc_limbs {
        // Below the packed trace's own arity overhead no trace exists, so
        // there is nothing to pair the limb group with.
        let Some(limb_physical) = field_inc_limbs.physical_num_vars(final_num_vars) else {
            return Ok(RegisteredRows::default());
        };
        mandatory.push(dense_precommit_profile(PolynomialGroupLayout::new(
            limb_physical,
            1,
        ))?);
    }
    for vars in direct_program_physical_vars {
        mandatory.push(dense_precommit_profile(PolynomialGroupLayout::new(
            *vars, 1,
        ))?);
    }
    let mut combinations = layouts.precommit_combinations()?;
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
                "unsupported one-hot K {other} for grouped advice provisioning"
            )))
        }
    };
    if !(min..=max).contains(&final_num_vars) {
        return Ok(RegisteredRows::default());
    }
    match one_hot_k {
        AKITA_ONE_HOT_K256 => provision::<JoltOneHotK256>(
            &combinations,
            honest_fold_policy_of::<JoltDenseBounded>(),
            [final_num_vars],
        ),
        AKITA_ONE_HOT_K16 => provision::<JoltOneHotK16>(
            &combinations,
            honest_fold_policy_of::<JoltDenseBounded>(),
            [final_num_vars],
        ),
        _ => unreachable!("one-hot K was validated above"),
    }
}

/// Drop every installed row. Tests only: production installs once per process.
#[cfg(test)]
pub fn reset_for_tests() {
    if let Ok(mut guard) = registry().write() {
        guard.clear();
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::configs::{JoltDenseBounded, JoltOneHotK256};
    use crate::schedules::emit;

    fn planned_profile<Cfg: akita_config::CommitmentConfig>(
        group: PolynomialGroupLayout,
    ) -> Result<GroupCommitPhaseParams, AkitaError> {
        let schedule = emit::regen::<Cfg>(group)?;
        GroupCommitPhaseParams::try_from_params(group, &schedule.root.params)
    }

    #[test]
    fn runtime_and_generated_dense_profiles_agree() {
        for physical_vars in emit::DENSE_NUM_VARS.0..=emit::DENSE_NUM_VARS.1 {
            let layout = PolynomialGroupLayout::new(physical_vars, 1);
            let from_catalog = dense_precommit_profile(layout).unwrap_or_else(|error| {
                panic!("dense catalog must cover {physical_vars} physical vars: {error}")
            });
            let from_planner = planned_profile::<JoltDenseBounded>(layout)
                .unwrap_or_else(|error| panic!("planner must solve {physical_vars} vars: {error}"));
            assert_eq!(
                from_catalog, from_planner,
                "dense profile for {physical_vars} vars diverges between catalog and planner"
            );
        }
    }

    #[test]
    fn provisioning_more_rows_than_the_cap_is_rejected() {
        let profile =
            dense_precommit_profile(PolynomialGroupLayout::new(emit::DENSE_NUM_VARS.0, 1)).unwrap();
        let error = provision::<JoltOneHotK256>(
            &[vec![profile]],
            honest_fold_policy_of::<JoltDenseBounded>(),
            0..=MAX_PROVISIONED_ROWS,
        )
        .expect_err("exceeding the row cap must be rejected");
        assert!(
            format!("{error}").contains("cap"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn independently_provisioned_profiles_share_the_process_cache() {
        reset_for_tests();
        let profile =
            dense_precommit_profile(PolynomialGroupLayout::new(emit::DENSE_NUM_VARS.0, 1)).unwrap();
        let row = plan_row::<JoltOneHotK256>(
            &AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(27, 1),
                precommitteds: vec![profile],
            },
            &[honest_fold_policy_of::<JoltDenseBounded>()],
        )
        .unwrap();
        let rows = |range: std::ops::Range<usize>| RegisteredRows {
            by_digest: range
                .map(|index| {
                    (
                        ScheduleRowDigest::from_bytes([u8::try_from(index).unwrap(); 32]),
                        row.clone(),
                    )
                })
                .collect(),
        };

        let _first = publish::<JoltOneHotK256>(rows(0..MAX_PROVISIONED_ROWS)).unwrap();
        let _second =
            publish::<JoltOneHotK256>(rows(MAX_PROVISIONED_ROWS..MAX_PROVISIONED_ROWS + 1))
                .expect("a second setup must not be rejected by rows cached for the first");
        reset_for_tests();
    }

    #[test]
    fn setup_params_provision_only_the_setup_final_arity() {
        reset_for_tests();
        let params = PrecommittedScheduleParams::new(
            None,
            Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
            FIXTURE_K16_FINAL_NUM_VARS.1,
        );
        let rows = params.provision(AKITA_ONE_HOT_K16).unwrap();
        assert_eq!(rows.rows().count(), 1);
        assert_eq!(
            rows.rows().next().unwrap().profiles().final_group.group,
            PolynomialGroupLayout::new(FIXTURE_K16_FINAL_NUM_VARS.1, 1)
        );
        reset_for_tests();
    }

    mod fixture {
        use super::*;
        use crate::configs::JoltOneHotK16;

        fn trusted_only(trusted: PolynomialGroupLayout) -> AdvicePrecommitLayouts {
            AdvicePrecommitLayouts {
                untrusted: None,
                trusted: Some(trusted),
            }
        }

        fn fixture_keys(
            final_num_vars: impl IntoIterator<Item = usize>,
        ) -> (GroupCommitPhaseParams, Vec<AkitaScheduleLookupKey>) {
            let profile = dense_precommit_profile(FIXTURE_TRUSTED_ADVICE_GROUP).unwrap();
            let keys = final_num_vars
                .into_iter()
                .map(|num_vars| AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(num_vars, 1),
                    precommitteds: vec![profile],
                })
                .collect();
            (profile, keys)
        }

        #[test]
        fn no_grouped_advice_row_is_cataloged_and_provisioning_plans_them_all() {
            reset_for_tests();
            let (_, keys) =
                fixture_keys(FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1);
            for key in &keys {
                assert!(
                    catalog_only_row::<JoltOneHotK16>(key).is_err(),
                    "a grouped advice row must never be checked in: {key:?}"
                );
            }

            let rows = provision_advice::<JoltOneHotK16>(
                trusted_only(FIXTURE_TRUSTED_ADVICE_GROUP),
                FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1,
            )
            .expect("provisioning must plan the whole range");
            assert_eq!(
                rows.rows().count(),
                keys.len(),
                "every final arity must be planned, since none is cataloged"
            );
            for key in &keys {
                let resolved = JoltOneHotK16::resolve_catalog_row_for_key(key)
                    .expect("a provisioned row must resolve through the hook");
                assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
            }
            reset_for_tests();
        }

        #[test]
        fn a_provisioned_row_resolves_through_every_hook() {
            reset_for_tests();
            let (profile, keys) = fixture_keys([27]);
            let key = keys.first().expect("one key");

            assert!(
                JoltOneHotK16::resolve_catalog_row_for_key(key).is_err(),
                "arity 27 must not be cataloged, or this test proves nothing"
            );

            let rows = provision::<JoltOneHotK16>(
                &[vec![profile]],
                honest_fold_policy_of::<JoltDenseBounded>(),
                [27],
            )
            .expect("provisioning an uncataloged key must plan it");
            assert_eq!(rows.rows().count(), 1);

            let by_key = JoltOneHotK16::resolve_catalog_row_for_key(key)
                .expect("the provisioned row must now resolve by key");
            let by_profiles = JoltOneHotK16::resolve_catalog_row_for_profiles(by_key.profiles())
                .expect("the provisioned row must resolve by exact profiles");
            let by_selection = JoltOneHotK16::resolve_schedule_selection(by_key.selection())
                .expect("the provisioned row must resolve by public selection");

            assert_eq!(by_key.selection(), by_profiles.selection());
            assert_eq!(by_key.selection(), by_selection.selection());
            reset_for_tests();
        }

        #[test]
        fn a_new_advice_capacity_provisions_and_resolves_every_final_arity() {
            reset_for_tests();
            let uncataloged_trusted =
                PolynomialGroupLayout::new(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars() + 1, 1);
            let range = FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1;
            let expected = range.clone().count();

            let rows =
                provision_advice::<JoltOneHotK16>(trusted_only(uncataloged_trusted), range.clone())
                    .expect("a previously unseen advice capacity must provision");
            assert_eq!(
                rows.rows().count(),
                expected,
                "every final arity at an uncataloged prefix must be planned"
            );

            let profile = dense_precommit_profile(uncataloged_trusted).unwrap();
            for num_vars in range {
                let key = AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(num_vars, 1),
                    precommitteds: vec![profile],
                };
                let row = JoltOneHotK16::resolve_catalog_row_for_key(&key)
                    .unwrap_or_else(|error| panic!("arity {num_vars} must resolve: {error}"));
                let by_selection = JoltOneHotK16::resolve_schedule_selection(row.selection())
                    .unwrap_or_else(|error| {
                        panic!("arity {num_vars} must resolve by selection: {error}")
                    });
                assert_eq!(by_selection.profiles(), row.profiles());
            }
            reset_for_tests();
        }

        #[test]
        fn republishing_an_identical_row_set_is_a_no_op() {
            reset_for_tests();
            let (profile, _) = fixture_keys([27]);
            let combinations = [vec![profile]];
            let policy = honest_fold_policy_of::<JoltDenseBounded>();
            let first = provision::<JoltOneHotK16>(&combinations, policy, [27])
                .expect("first provisioning");
            let second = provision::<JoltOneHotK16>(&combinations, policy, [27])
                .expect("re-provisioning an identical set must succeed");
            assert_eq!(first.rows().count(), second.rows().count());
            for row in first.rows() {
                assert!(second.by_selection(row.selection()).is_some());
            }
            reset_for_tests();
        }

        #[test]
        fn two_advice_capacities_coexist_without_aliasing() {
            reset_for_tests();
            let small = FIXTURE_TRUSTED_ADVICE_GROUP;
            let large = PolynomialGroupLayout::new(small.num_vars() + 1, 1);

            let small_rows =
                provision_advice::<JoltOneHotK16>(trusted_only(small), [27]).expect("small");
            let large_rows =
                provision_advice::<JoltOneHotK16>(trusted_only(large), [27]).expect("large");
            for (layout, own) in [(small, &small_rows), (large, &large_rows)] {
                let profile = dense_precommit_profile(layout).unwrap();
                let key = AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(27, 1),
                    precommitteds: vec![profile],
                };
                let resolved = JoltOneHotK16::resolve_catalog_row_for_key(&key)
                    .unwrap_or_else(|error| panic!("{layout:?} must still resolve: {error}"));
                assert_eq!(
                    resolved.profiles().precommitteds,
                    vec![profile],
                    "resolved row must carry its own prefix, not the other capacity's"
                );
                assert!(
                    own.rows()
                        .any(|row| row.selection() == resolved.selection()),
                    "each capacity must resolve to a row from its own set"
                );
            }
            reset_for_tests();
        }

        #[test]
        fn a_digest_collision_with_different_profiles_is_rejected() {
            reset_for_tests();
            let (profile, _) = fixture_keys([27]);
            let mut rows = provision::<JoltOneHotK16>(
                &[vec![profile]],
                honest_fold_policy_of::<JoltDenseBounded>(),
                [27],
            )
            .expect("provisioning must succeed");

            let other = plan_row::<JoltOneHotK16>(
                &AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(28, 1),
                    precommitteds: vec![profile],
                },
                &[honest_fold_policy_of::<JoltDenseBounded>()],
            )
            .expect("plan a second row");
            let stolen_digest = rows.by_digest.keys().next().copied().expect("one row");
            rows.by_digest.clear();
            let _ = rows.by_digest.insert(stolen_digest, other);

            let error =
                publish::<JoltOneHotK16>(rows).expect_err("a digest collision must be rejected");
            assert!(
                format!("{error}").contains("collision"),
                "unexpected error: {error}"
            );
            reset_for_tests();
        }

        #[cfg(feature = "field-inline")]
        mod field_inc_limbs {
            use jolt_claims::lattice::MIN_DENSE_OBJECT_NUM_VARS;
            use jolt_claims::protocols::field_inline::lattice::{
                field_inc_limb_count, FieldIncLimbPackingPlan, FieldIncLimbShape,
            };
            use jolt_claims::protocols::jolt::lattice::packing::one_hot_trace_column_capacity;

            use super::*;
            use crate::adapters::AkitaField;
            use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

            /// The packed trace's arity overhead over its own `log_T`: the
            /// chunk plus selector variables, constant per K.
            fn trace_arity_overhead(one_hot_k: usize) -> usize {
                let log_k_chunk = one_hot_k.ilog2() as usize;
                log_k_chunk + one_hot_trace_column_capacity(log_k_chunk).unwrap().ilog2() as usize
            }

            /// The production caller's derivation of the FR arity line, from
            /// the jolt-claims laws: the packed trace's arity overhead over
            /// `log_T` and the limb plan's floor/selector geometry.
            fn law_derived_params(one_hot_k: usize) -> FieldIncLimbScheduleParams {
                let limbs = field_inc_limb_count::<AkitaField>();
                FieldIncLimbScheduleParams::new(
                    trace_arity_overhead(one_hot_k),
                    MIN_DENSE_OBJECT_NUM_VARS,
                    limbs.next_power_of_two().ilog2() as usize,
                )
            }

            fn limb_profile(
                params: FieldIncLimbScheduleParams,
                final_num_vars: usize,
            ) -> GroupCommitPhaseParams {
                dense_precommit_profile(PolynomialGroupLayout::new(
                    params.physical_num_vars(final_num_vars).unwrap(),
                    1,
                ))
                .unwrap()
            }

            /// The registry's carried arity line must equal the jolt-claims
            /// packing law at every final arity, in both K regimes.
            #[test]
            fn carried_arity_line_matches_the_packing_law() {
                let limbs = field_inc_limb_count::<AkitaField>();
                assert_eq!(limbs, 2, "fp128 decomposes into two u64 limbs");
                for (one_hot_k, (min, max)) in [
                    (AKITA_ONE_HOT_K16, K16_NUM_VARS),
                    (AKITA_ONE_HOT_K256, K256_NUM_VARS),
                ] {
                    let params = law_derived_params(one_hot_k);
                    for final_num_vars in min..=max {
                        let carried = params.physical_num_vars(final_num_vars);
                        let expected = final_num_vars
                            .checked_sub(trace_arity_overhead(one_hot_k))
                            .map(|log_t| {
                                FieldIncLimbPackingPlan::new(&FieldIncLimbShape { limbs, log_t })
                                    .unwrap()
                                    .packing()
                                    .packed_num_vars()
                            });
                        assert_eq!(
                            carried, expected,
                            "K={one_hot_k} final arity {final_num_vars}: carried arity diverges \
                             from the packing law"
                        );
                    }
                }
            }

            /// The prover pads packed traces to `MIN_PADDED_TRACE_LENGTH`
            /// (jolt-prover, `1 << 12` on akita builds), so the smallest
            /// reachable FR final arity is `overhead + 12`.
            const PROVER_MIN_LOG_T: usize = 12;

            /// Every reachable final arity of the K catalog provisions its own
            /// FR row (production provisions the setup's single final arity)
            /// that resolves through the hook. Doubles as the norm-budget
            /// check: the rows plan under the same u64-bounded dense fold
            /// policy advice uses, so a planned row means the limb words fit
            /// that budget. Arities below the prover's trace floor are
            /// unreachable; whether the planner admits them is not asserted.
            fn fr_rows_plan_and_resolve_at_every_arity<Cfg: CommitmentConfig + 'static>(
                one_hot_k: usize,
                (declared_min, ceiling): (usize, usize),
            ) {
                reset_for_tests();
                let params = law_derived_params(one_hot_k);
                let reachable_min = trace_arity_overhead(one_hot_k) + PROVER_MIN_LOG_T;
                for final_num_vars in declared_min..=ceiling {
                    let rows = PrecommittedScheduleParams::new(None, None, final_num_vars)
                        .with_field_inc_limbs(params)
                        .provision(one_hot_k)
                        .expect("FR provisioning must plan or skip every arity");
                    assert!(
                        rows.rows().count() <= 1,
                        "K={one_hot_k} final arity {final_num_vars}: one FR row at most"
                    );
                    if final_num_vars < reachable_min {
                        continue;
                    }
                    assert_eq!(
                        rows.rows().count(),
                        1,
                        "K={one_hot_k} final arity {final_num_vars} must plan its FR row"
                    );
                    let key = AkitaScheduleLookupKey {
                        final_group: PolynomialGroupLayout::new(final_num_vars, 1),
                        precommitteds: vec![limb_profile(params, final_num_vars)],
                    };
                    let resolved = Cfg::resolve_catalog_row_for_key(&key).unwrap_or_else(|error| {
                        panic!(
                            "K={one_hot_k} final arity {final_num_vars} must resolve its FR row: \
                             {error}"
                        )
                    });
                    assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
                }
                reset_for_tests();
            }

            #[test]
            fn fr_rows_plan_and_resolve_at_every_k16_arity() {
                fr_rows_plan_and_resolve_at_every_arity::<JoltOneHotK16>(
                    AKITA_ONE_HOT_K16,
                    K16_NUM_VARS,
                );
            }

            #[test]
            fn fr_rows_plan_and_resolve_at_every_k256_arity() {
                fr_rows_plan_and_resolve_at_every_arity::<JoltOneHotK256>(
                    AKITA_ONE_HOT_K256,
                    K256_NUM_VARS,
                );
            }

            /// With both advice kinds declared, every advice presence
            /// combination is provisioned with the FR profile as its last
            /// group and no FR-absent row exists: an FR-on prover commits the
            /// group on every proof, so none is constructible.
            #[test]
            fn fr_rows_append_the_limb_group_to_every_advice_combination() {
                reset_for_tests();
                let params = law_derived_params(AKITA_ONE_HOT_K16);
                let final_num_vars = FIXTURE_K16_FINAL_NUM_VARS.1;
                let trusted = FIXTURE_TRUSTED_ADVICE_GROUP.num_vars();
                let rows = PrecommittedScheduleParams::new(
                    Some(trusted + 1),
                    Some(trusted),
                    final_num_vars,
                )
                .with_field_inc_limbs(params)
                .provision(AKITA_ONE_HOT_K16)
                .expect("FR-composed provisioning must plan every combination");
                assert_eq!(rows.rows().count(), 4);
                let limb = limb_profile(params, final_num_vars);
                for row in rows.rows() {
                    assert_eq!(row.profiles().precommitteds.last(), Some(&limb));
                }
                reset_for_tests();
            }
        }

        #[test]
        fn setup_capacity_covers_a_provisioned_row() {
            reset_for_tests();
            let (profile, _) = fixture_keys([27]);
            let baseline = JoltOneHotK16::setup_matrix_capacity(27, 2)
                .expect("baseline capacity")
                .num_field_elements;
            let _ = provision::<JoltOneHotK16>(
                &[vec![profile]],
                honest_fold_policy_of::<JoltDenseBounded>(),
                [27],
            )
            .expect("provisioning must succeed");
            let provisioned = JoltOneHotK16::setup_matrix_capacity(27, 2)
                .expect("capacity after provisioning")
                .num_field_elements;
            assert!(
                provisioned >= baseline,
                "provisioned capacity {provisioned} must cover the baseline {baseline}"
            );
            reset_for_tests();
        }
    }
}
