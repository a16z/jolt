//! Canonical schedule planning for Jolt's Akita configurations.

use akita_config::{policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::find_schedule;
use akita_types::{AkitaScheduleLookupKey, FoldSchedule};

pub(crate) fn plan_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    let planned = find_schedule(
        key,
        Cfg::committed_source_contract()?,
        &[],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
