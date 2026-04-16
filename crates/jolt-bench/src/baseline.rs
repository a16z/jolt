use std::fs;
use std::path::Path;

use crate::output::{BenchReport, Run};

#[derive(Clone, Debug)]
pub struct Regression {
    pub metric: &'static str,
    pub baseline: f64,
    pub observed: f64,
    pub ratio: f64,
}

fn find_stack<'a>(report: &'a BenchReport, stack: &str) -> Option<&'a Run> {
    report.runs.iter().find(|r| r.stack == stack)
}

fn compare_metric(
    regressions: &mut Vec<Regression>,
    metric: &'static str,
    baseline: Option<f64>,
    observed: Option<f64>,
    threshold: f64,
) {
    let (Some(b), Some(o)) = (baseline, observed) else {
        return;
    };
    if b <= 0.0 {
        return;
    }
    let ratio = o / b;
    if ratio > threshold {
        regressions.push(Regression {
            metric,
            baseline: b,
            observed: o,
            ratio,
        });
    }
}

/// Compare the current report's `modular` row against the baseline's `core`
/// row. Returns the list of metrics exceeding `threshold` × baseline.
pub fn check_regressions(
    baseline: &BenchReport,
    current: &BenchReport,
    threshold: f64,
) -> Vec<Regression> {
    let mut out = Vec::new();
    let Some(baseline_core) = find_stack(baseline, "core") else {
        return out;
    };
    let Some(current_modular) = find_stack(current, "modular") else {
        return out;
    };
    if current_modular.unsupported {
        return out;
    }

    compare_metric(
        &mut out,
        "prove_ms",
        baseline_core.prove_ms,
        current_modular.prove_ms,
        threshold,
    );
    compare_metric(
        &mut out,
        "verify_ms",
        baseline_core.verify_ms,
        current_modular.verify_ms,
        threshold,
    );
    compare_metric(
        &mut out,
        "peak_rss_mb",
        baseline_core.peak_rss_mb.map(|v| v as f64),
        current_modular.peak_rss_mb.map(|v| v as f64),
        threshold,
    );
    compare_metric(
        &mut out,
        "proof_bytes",
        baseline_core.proof_bytes.map(|v| v as f64),
        current_modular.proof_bytes.map(|v| v as f64),
        threshold,
    );
    out
}

pub fn load_baseline(path: &Path) -> BenchReport {
    let bytes = fs::read(path).expect("read baseline file");
    serde_json::from_slice(&bytes).expect("parse baseline JSON")
}
