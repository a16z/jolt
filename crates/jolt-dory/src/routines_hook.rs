//! Device-tier injection seam for the [`crate::routines`] vector ops that
//! dominate dory-pcs's reduce-and-fold rounds: the two uniform-scalar folds
//! (`vs[i] += s·bases[i]`, `vs[i] = s·vs[i] + addends[i]`, both `out[i] =
//! s·P[i] + Q[i]` at group level) and the VMV preamble's fixed-base sweep
//! (`out[i] = base·scalars[i]`).
//!
//! Like [`crate::hint_hook`], this is an acceleration seam, never a
//! semantics seam: an installed hook may serve a call (`Some(out)` — the
//! same group elements the CPU path computes, possibly in a different
//! Jacobian representative) or decline it (`None` — undersized work, dead
//! device), and the routine falls through to its own arithmetic on `None`.
//! Every consumer of the folded vectors (pairings, MSMs, later folds, the
//! serialized final message) normalizes before use, so group equality is
//! byte equality for the proof. The guard returned by
//! [`install_routine_hooks`] clears the seam on drop; installers scope it
//! to one proof (the jolt backend parks it in the proof session).
//! Concurrent proofs in one process may race the slot — the loser silently
//! computes on the CPU, which is always correct.
//!
//! `jolt-dory` cannot depend on the device crates (they depend on it), so
//! the seam is bare `fn` pointers injected from above.

use std::sync::{PoisonError, RwLock};

use ark_bn254::{Fr as ArkworksFr, G1Projective, G2Projective};

/// `out[i] = s·ps[i] + qs[i]` over G1: `Some` when served (group-equal to
/// the CPU path), `None` to decline.
pub type G1ScalarMulAddFn =
    fn(&[G1Projective], &[G1Projective], &ArkworksFr) -> Option<Vec<G1Projective>>;

/// `out[i] = s·ps[i] + qs[i]` over G2.
pub type G2ScalarMulAddFn =
    fn(&[G2Projective], &[G2Projective], &ArkworksFr) -> Option<Vec<G2Projective>>;

/// `out[i] = base·scalars[i]` over G2 (one shared base).
pub type G2FixedBaseMulFn = fn(&G2Projective, &[ArkworksFr]) -> Option<Vec<G2Projective>>;

/// `Σ scalars[i]·bases[i]` over G1 (the VMV preamble's variable-base MSMs).
pub type G1MsmFn = fn(&[G1Projective], &[ArkworksFr]) -> Option<G1Projective>;

/// The routine implementations a device backend installs together.
#[derive(Clone, Copy)]
pub struct RoutineHooks {
    pub g1_scalar_mul_add: G1ScalarMulAddFn,
    pub g2_scalar_mul_add: G2ScalarMulAddFn,
    pub g2_fixed_base_mul: G2FixedBaseMulFn,
    pub g1_msm: G1MsmFn,
}

static HOOKS: RwLock<Option<RoutineHooks>> = RwLock::new(None);

/// Clears the hooks installed by [`install_routine_hooks`] on drop.
#[must_use = "dropping the guard immediately uninstalls the hooks"]
pub struct RoutineHooksGuard(());

impl Drop for RoutineHooksGuard {
    fn drop(&mut self) {
        *HOOKS.write().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// Install `hooks` for the [`crate::routines`] vector ops until the
/// returned guard drops, replacing any previous installation.
pub fn install_routine_hooks(hooks: RoutineHooks) -> RoutineHooksGuard {
    *HOOKS.write().unwrap_or_else(PoisonError::into_inner) = Some(hooks);
    RoutineHooksGuard(())
}

/// The currently installed hooks, if any.
pub(crate) fn routine_hooks() -> Option<RoutineHooks> {
    *HOOKS.read().unwrap_or_else(PoisonError::into_inner)
}
