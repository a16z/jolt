use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq)]
pub struct StageTiming {
    pub label: &'static str,
    pub time_ms: f64,
}

thread_local! {
    static STAGE_TIMINGS: RefCell<Vec<StageTiming>> = const { RefCell::new(Vec::new()) };
}

pub fn reset_stage_timings() {
    STAGE_TIMINGS.with(|timings| timings.borrow_mut().clear());
}

pub fn take_stage_timings() -> Vec<StageTiming> {
    STAGE_TIMINGS.with(|timings| std::mem::take(&mut *timings.borrow_mut()))
}

pub(crate) fn record_stage_timing(label: &'static str, time_ms: f64) {
    STAGE_TIMINGS.with(|timings| {
        timings.borrow_mut().push(StageTiming { label, time_ms });
    });
}
