//! Canonical schedule planning for Jolt's Akita configurations.

use akita_config::{policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::{find_schedule, PlannerPolicy};
use akita_types::sis::CommittedSourceContract;
use akita_types::{AkitaScheduleLookupKey, FoldSchedule};

pub(crate) fn plan_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    auxiliary_source_contracts: &[CommittedSourceContract],
) -> Result<FoldSchedule, AkitaError> {
    plan_schedule_with_policy::<Cfg>(key, auxiliary_source_contracts, &policy_of::<Cfg>())
}

pub(crate) fn plan_schedule_with_policy<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    auxiliary_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
) -> Result<FoldSchedule, AkitaError> {
    let planned = find_schedule(
        key,
        Cfg::committed_source_contract()?,
        auxiliary_source_contracts,
        policy,
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
