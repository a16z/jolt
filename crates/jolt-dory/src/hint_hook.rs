//! Device-tier injection seam for `DoryScheme::combine_hints`: a
//! process-global, guard-scoped hook a device backend installs around a
//! proof so the stage-8 batch opening's hint combination (per row an
//! independent multi-scalar multiplication over the ragged hint matrix) can
//! run off the CPU.
//!
//! The hook is an acceleration seam, never a semantics seam: an installed
//! hook may serve a call (`Some(combined)` — the same group elements the CPU
//! path computes) or decline it (`None` — undersized work, dead device), and
//! `combine_hints` falls through to its own arithmetic on `None`. Because
//! results agree at group level, an install can only change WHERE the
//! combination runs. The guard returned by [`install_combine_hints_hook`]
//! clears the hook on drop; installers scope it to one proof (the jolt
//! backend parks it in the proof session). Concurrent proofs in one process
//! with different backends may race the slot — the loser silently computes
//! on the CPU, which is always correct.
//!
//! `jolt-dory` cannot depend on the device crates (they depend on it), so
//! the seam is a bare `fn` pointer injected from above.

use std::sync::{PoisonError, RwLock};

use jolt_field::Fr;

use crate::types::DoryHint;

/// A `combine_hints` implementation candidate: `Some` when the call was
/// served (value-equal to the CPU path at group level), `None` to decline.
pub type CombineHintsFn = fn(&[DoryHint], &[Fr]) -> Option<DoryHint>;

static HOOK: RwLock<Option<CombineHintsFn>> = RwLock::new(None);

/// Clears the hook installed by [`install_combine_hints_hook`] on drop.
#[must_use = "dropping the guard immediately uninstalls the hook"]
pub struct CombineHintsHookGuard(());

impl Drop for CombineHintsHookGuard {
    fn drop(&mut self) {
        *HOOK.write().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// Install `hook` for `DoryScheme::combine_hints` calls until the returned
/// guard drops, replacing any previous installation.
pub fn install_combine_hints_hook(hook: CombineHintsFn) -> CombineHintsHookGuard {
    *HOOK.write().unwrap_or_else(PoisonError::into_inner) = Some(hook);
    CombineHintsHookGuard(())
}

/// The currently installed hook, if any.
pub(crate) fn combine_hints_hook() -> Option<CombineHintsFn> {
    *HOOK.read().unwrap_or_else(PoisonError::into_inner)
}
