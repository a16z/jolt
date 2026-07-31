//! Scoped device-resident implementation for the transparent Dory reduce loop.
//!
//! The vendored protocol driver owns transcript order and the host tail. This
//! seam only supplies value-exact group operations over state retained by the
//! backend between rounds.

#![allow(missing_docs)]

use std::sync::{PoisonError, RwLock};

use crate::primitives::arithmetic::ResidentRoundHooks;

use super::BN254;

pub type Hooks = ResidentRoundHooks<BN254>;

static HOOKS: RwLock<Option<Hooks>> = RwLock::new(None);

#[must_use = "dropping the guard immediately uninstalls the hook"]
pub struct ResidentRoundHookGuard(());

impl Drop for ResidentRoundHookGuard {
    fn drop(&mut self) {
        *HOOKS.write().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

pub fn install_resident_round_hook(hooks: Hooks) -> ResidentRoundHookGuard {
    *HOOKS.write().unwrap_or_else(PoisonError::into_inner) = Some(hooks);
    ResidentRoundHookGuard(())
}

pub(crate) fn resident_round_hooks() -> Option<Hooks> {
    *HOOKS.read().unwrap_or_else(PoisonError::into_inner)
}
