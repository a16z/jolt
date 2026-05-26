use serde::{Deserialize, Serialize};

use crate::metrics::RunMetrics;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerfGate {
    pub warn_ratio: f64,
    pub fail_ratio: f64,
    pub min_samples: u32,
    pub confirmation_size: Option<u64>,
}

impl PerfGate {
    pub const fn canonical_frontier() -> Self {
        Self {
            warn_ratio: 1.05,
            fail_ratio: 1.10,
            min_samples: 3,
            confirmation_size: Some(1_048_576),
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if !self.warn_ratio.is_finite() || !self.fail_ratio.is_finite() {
            return Err("perf gate ratios must be finite");
        }
        if self.warn_ratio < 1.0 {
            return Err("perf gate warn ratio must be at least 1.0");
        }
        if self.fail_ratio <= self.warn_ratio {
            return Err("perf gate fail ratio must be greater than warn ratio");
        }
        if self.min_samples == 0 {
            return Err("perf gate must require at least one sample");
        }
        if matches!(self.confirmation_size, Some(0)) {
            return Err("perf gate confirmation size must be nonzero when present");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
    NotMeasured,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerfEvaluation {
    pub status: GateStatus,
    pub time_ratio: Option<f64>,
    pub peak_rss_ratio: Option<f64>,
}

pub fn evaluate_perf(gate: PerfGate, core: &RunMetrics, modular: &RunMetrics) -> PerfEvaluation {
    let time_ratio = ratio(modular.time_ms, core.time_ms);
    let peak_rss_ratio = ratio_u64(modular.peak_rss_bytes, core.peak_rss_bytes);
    let status = [time_ratio, peak_rss_ratio]
        .into_iter()
        .fold(GateStatus::NotMeasured, |status, ratio| {
            combine_status(status, ratio_status(gate, ratio))
        });

    PerfEvaluation {
        status,
        time_ratio,
        peak_rss_ratio,
    }
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0.0 => Some(numerator / denominator),
        _ => None,
    }
}

fn ratio_u64(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0 => {
            Some(numerator as f64 / denominator as f64)
        }
        _ => None,
    }
}

fn ratio_status(gate: PerfGate, ratio: Option<f64>) -> GateStatus {
    match ratio {
        Some(ratio) if ratio > gate.fail_ratio => GateStatus::Fail,
        Some(ratio) if ratio > gate.warn_ratio => GateStatus::Warn,
        Some(_) => GateStatus::Pass,
        None => GateStatus::NotMeasured,
    }
}

fn combine_status(left: GateStatus, right: GateStatus) -> GateStatus {
    match (left, right) {
        (GateStatus::Fail, _) | (_, GateStatus::Fail) => GateStatus::Fail,
        (GateStatus::Warn, _) | (_, GateStatus::Warn) => GateStatus::Warn,
        (GateStatus::Pass, _) | (_, GateStatus::Pass) => GateStatus::Pass,
        (GateStatus::NotMeasured, GateStatus::NotMeasured) => GateStatus::NotMeasured,
    }
}
