//! Canonical schedule planning for Jolt's Akita configurations.

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::find_schedule;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{AkitaScheduleLookupKey, FoldSchedule};

pub(crate) fn plan_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
) -> Result<FoldSchedule, AkitaError> {
    let planned = find_schedule(
        key,
        honest_fold_policy_of::<Cfg>(),
        precommitted_honest_fold_policies,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
