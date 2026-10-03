use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum FsScope {
    #[default]
    Preamble,
    Commitments,
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6a,
    Stage6b,
    Stage7,
    Stage8,
    BlindFold,
}

thread_local! {
    static CURRENT_SCOPE: Cell<FsScope> = const { Cell::new(FsScope::Preamble) };
}

pub struct FsScopeGuard {
    previous: FsScope,
}

impl Drop for FsScopeGuard {
    fn drop(&mut self) {
        CURRENT_SCOPE.set(self.previous);
    }
}

#[must_use]
pub fn enter(scope: FsScope) -> FsScopeGuard {
    let previous = CURRENT_SCOPE.replace(scope);
    FsScopeGuard { previous }
}

pub fn current() -> FsScope {
    CURRENT_SCOPE.get()
}
