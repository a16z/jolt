//! Test-only Fiat-Shamir verifier scopes.

use std::cell::Cell;

/// Verifier region active when a transcript challenge is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum FsScope {
    /// Input validation and transcript preamble.
    #[default]
    Preamble,
    /// Proof and preprocessing commitments.
    Commitments,
    /// Stage 1.
    Stage1,
    /// Stage 2.
    Stage2,
    /// Stage 3.
    Stage3,
    /// Stage 4.
    Stage4,
    /// Stage 5.
    Stage5,
    /// Stage 6 address phase.
    Stage6a,
    /// Stage 6 cycle phase.
    Stage6b,
    /// Stage 7.
    Stage7,
    /// Akita reconstruction.
    Reconstruction,
    /// Final opening checks.
    Stage8,
    /// BlindFold verification.
    BlindFold,
}

thread_local! {
    static CURRENT_SCOPE: Cell<FsScope> = const { Cell::new(FsScope::Preamble) };
}

/// Restores the previous verifier scope on drop.
pub struct FsScopeGuard {
    previous: FsScope,
}

impl Drop for FsScopeGuard {
    fn drop(&mut self) {
        CURRENT_SCOPE.set(self.previous);
    }
}

/// Marks subsequent transcript operations as belonging to `scope`.
#[must_use]
pub fn enter(scope: FsScope) -> FsScopeGuard {
    let previous = CURRENT_SCOPE.replace(scope);
    FsScopeGuard { previous }
}

/// Returns the verifier scope active on this thread.
pub fn current() -> FsScope {
    CURRENT_SCOPE.get()
}
