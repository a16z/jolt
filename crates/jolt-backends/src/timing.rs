use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
pub struct BackendTiming {
    pub label: &'static str,
    pub time_ms: f64,
}

static BACKEND_TIMINGS: Mutex<Vec<BackendTiming>> = Mutex::new(Vec::new());

pub fn reset_backend_timings() {
    if let Ok(mut timings) = BACKEND_TIMINGS.lock() {
        timings.clear();
    }
}

pub fn take_backend_timings() -> Vec<BackendTiming> {
    BACKEND_TIMINGS
        .lock()
        .map(|mut timings| std::mem::take(&mut *timings))
        .unwrap_or_default()
}

pub(crate) fn record_backend_timing(label: &'static str, time_ms: f64) {
    if let Ok(mut timings) = BACKEND_TIMINGS.lock() {
        timings.push(BackendTiming { label, time_ms });
    }
}
