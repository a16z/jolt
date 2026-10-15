//! Canonical schedule planning for Jolt's Akita configurations.

use akita_config::{policy_of, CommitmentConfig};
use akita_params::sis::CommittedSourceContract;
use akita_params::{FoldSchedule, ScheduleLookupKey};
use akita_pcs::AkitaError;
use akita_planner::find_schedule;

pub(crate) fn plan_schedule<Cfg: CommitmentConfig>(
    key: &ScheduleLookupKey,
    auxiliary_source_contracts: &[CommittedSourceContract],
) -> Result<FoldSchedule, AkitaError> {
    let planned = find_schedule(
        key,
        Cfg::committed_source_contract()?,
        auxiliary_source_contracts,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
