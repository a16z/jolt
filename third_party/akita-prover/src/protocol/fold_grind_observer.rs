//! Prover-side fold grind probe metrics (profile / diagnostics only).

use std::cell::RefCell;

/// One fold-level grind outcome recorded during proving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldGrindObservation {
    /// Wire nonce committed into the proof.
    pub grind_nonce: u32,
    /// Number of off-sponge probes before acceptance (includes the winner).
    pub grind_probe_count: u32,
}

struct ObserverState {
    active: bool,
    records: Vec<FoldGrindObservation>,
}

thread_local! {
    static FOLD_GRIND_OBSERVER: RefCell<ObserverState> = const {
        RefCell::new(ObserverState {
            active: false,
            records: Vec::new(),
        })
    };
}

/// RAII guard that activates fold-grind observation on the current thread.
pub struct FoldGrindObserverGuard;

impl FoldGrindObserverGuard {
    /// Begin recording fold grind probe counts for subsequent prove calls.
    pub fn install() -> Self {
        FOLD_GRIND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = true;
            state.records.clear();
        });
        Self
    }

    /// Drain recorded observations and deactivate the observer.
    pub fn take() -> Vec<FoldGrindObservation> {
        FOLD_GRIND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = false;
            std::mem::take(&mut state.records)
        })
    }
}

impl Drop for FoldGrindObserverGuard {
    fn drop(&mut self) {
        FOLD_GRIND_OBSERVER.with(|cell| {
            cell.borrow_mut().active = false;
        });
    }
}

pub(crate) fn record_fold_grind_acceptance(grind_nonce: u32, grind_probe_count: u32) {
    debug_assert!(grind_probe_count > 0, "grind probe count must be positive");
    FOLD_GRIND_OBSERVER.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.active {
            state.records.push(FoldGrindObservation {
                grind_nonce,
                grind_probe_count,
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_take_roundtrip_records_probe_metrics() {
        let _guard = FoldGrindObserverGuard::install();
        record_fold_grind_acceptance(7, 3);
        record_fold_grind_acceptance(0, 1);
        let records = FoldGrindObserverGuard::take();
        assert_eq!(
            records,
            vec![
                FoldGrindObservation {
                    grind_nonce: 7,
                    grind_probe_count: 3,
                },
                FoldGrindObservation {
                    grind_nonce: 0,
                    grind_probe_count: 1,
                },
            ]
        );
    }

    #[test]
    fn inactive_observer_drops_records() {
        record_fold_grind_acceptance(1, 1);
        assert!(FoldGrindObserverGuard::take().is_empty());
    }
}
