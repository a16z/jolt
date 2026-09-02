//! Device-tier injection seam for the three multi-pairing entry points
//! (Jolt Metal Miller lane).
//!
//! An acceleration seam, never a semantics seam: an installed hook may
//! serve a call (`Some(gt)` — the exact GT the CPU path computes; Miller
//! products and final exponentiation are exact field arithmetic, so any
//! correct evaluator yields identical bytes) or decline it (`None` —
//! undersized work, dead device), and the pairing falls through to the
//! optimized CPU path. The guard returned by [`install_multi_pair_hook`]
//! clears the seam on drop; installers scope it to one proof. Concurrent
//! installs race the slot — the loser computes on the CPU, always correct.

#![allow(missing_docs)]

use std::sync::{PoisonError, RwLock};

use super::ark_group::{ArkG1, ArkG2, ArkGT};

/// The full multi-pairing `Π e(ps[i], qs[i])`: `Some` when served, `None`
/// to decline. Inputs may include identity points (skipped, as the CPU
/// path's pair filter does).
pub type MultiPairFn = fn(&[ArkG1], &[ArkG2]) -> Option<ArkGT>;

static HOOK: RwLock<Option<MultiPairFn>> = RwLock::new(None);

/// Clears the hook installed by [`install_multi_pair_hook`] on drop.
#[must_use = "dropping the guard immediately uninstalls the hook"]
pub struct MultiPairHookGuard(());

impl Drop for MultiPairHookGuard {
    fn drop(&mut self) {
        *HOOK.write().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// Install `hook` for `multi_pair` / `multi_pair_g2_setup` /
/// `multi_pair_g1_setup` until the returned guard drops, replacing any
/// previous installation.
pub fn install_multi_pair_hook(hook: MultiPairFn) -> MultiPairHookGuard {
    *HOOK.write().unwrap_or_else(PoisonError::into_inner) = Some(hook);
    MultiPairHookGuard(())
}

/// The currently installed hook, if any.
pub(crate) fn multi_pair_hook() -> Option<MultiPairFn> {
    *HOOK.read().unwrap_or_else(PoisonError::into_inner)
}
