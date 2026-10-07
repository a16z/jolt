//! Canonical schedule planning for Jolt's Akita configurations.

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_pcs::AkitaError;
use akita_planner::{find_schedule, find_schedule_with_root_shape, RootShape};
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{AkitaScheduleLookupKey, FoldSchedule};

/// `root_shape` fixes the root final group's geometry; every other root and
/// suffix choice stays the planner's.
pub(crate) fn plan_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    root_shape: Option<RootShape>,
) -> Result<FoldSchedule, AkitaError> {
    let planned = match root_shape {
        None => find_schedule(
            key,
            honest_fold_policy_of::<Cfg>(),
            precommitted_honest_fold_policies,
            &policy_of::<Cfg>(),
            Cfg::ring_challenge_config,
        ),
        Some(shape) => find_schedule_with_root_shape(
            key,
            honest_fold_policy_of::<Cfg>(),
            precommitted_honest_fold_policies,
            &policy_of::<Cfg>(),
            Cfg::ring_challenge_config,
            shape,
        ),
    }?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}
