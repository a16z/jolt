//! Core-vs-Bolt performance gate primitives.
//!
//! This module intentionally does not know how to run a specific protocol or
//! workload. Harnesses provide measured metrics and observed span names; this
//! module owns the shared comparison contract.

use std::fmt;

/// Required top-level span names for Jolt-on-Bolt perf gates.
///
/// Harnesses may emit more detailed spans beneath these, but this set should
/// stay stable so Perfetto traces and CI reports can compare core and Bolt runs.
pub const CORE_VS_BOLT_REQUIRED_SPANS: &[&str] = &[
    "core.setup",
    "core.prove",
    "core.verify",
    "bolt.setup",
    "bolt.commitment",
    "bolt.stage1",
    "bolt.stage2",
    "bolt.stage3",
    "bolt.stage4",
    "bolt.stage5",
    "bolt.stage6",
    "bolt.stage7",
    "bolt.stage8",
    "bolt.evaluate",
    "bolt.verify",
];

/// Path measured by a paired core-vs-Bolt perf gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfPath {
    Core,
    Bolt,
}

impl PerfPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Bolt => "bolt",
        }
    }
}

impl fmt::Display for PerfPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metric compared by a core-vs-Bolt perf gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerfMetric {
    SetupMs,
    ProveMs,
    VerifyMs,
    ProofBytes,
    PeakRssMb,
}

impl PerfMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SetupMs => "setup_ms",
            Self::ProveMs => "prove_ms",
            Self::VerifyMs => "verify_ms",
            Self::ProofBytes => "proof_bytes",
            Self::PeakRssMb => "peak_rss_mb",
        }
    }
}

impl fmt::Display for PerfMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Median metrics for one side of a paired perf run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerfMetrics {
    pub setup_ms: Option<f64>,
    pub prove_ms: Option<f64>,
    pub verify_ms: Option<f64>,
    pub proof_bytes: Option<u64>,
    pub peak_rss_mb: Option<u64>,
    pub span_names: Vec<String>,
}

impl PerfMetrics {
    pub fn with_span_names<I, S>(mut self, span_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.span_names = span_names.into_iter().map(Into::into).collect();
        self
    }
}

/// Ratio thresholds for comparing Bolt metrics to core metrics.
///
/// A `None` threshold disables that metric. The default checks prove, verify,
/// proof-size, and peak-RSS regressions and leaves setup informational unless a
/// harness opts in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerfGateThresholds {
    pub max_setup_ratio: Option<f64>,
    pub max_prove_ratio: Option<f64>,
    pub max_verify_ratio: Option<f64>,
    pub max_proof_size_ratio: Option<f64>,
    pub max_peak_rss_ratio: Option<f64>,
}

impl PerfGateThresholds {
    pub const fn disabled() -> Self {
        Self {
            max_setup_ratio: None,
            max_prove_ratio: None,
            max_verify_ratio: None,
            max_proof_size_ratio: None,
            max_peak_rss_ratio: None,
        }
    }

    pub const fn five_percent() -> Self {
        Self {
            max_setup_ratio: None,
            max_prove_ratio: Some(1.05),
            max_verify_ratio: Some(1.05),
            max_proof_size_ratio: Some(1.05),
            max_peak_rss_ratio: Some(1.05),
        }
    }
}

impl Default for PerfGateThresholds {
    fn default() -> Self {
        Self::five_percent()
    }
}

/// Successful metric comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricRatio {
    pub metric: PerfMetric,
    pub reference: f64,
    pub candidate: f64,
    pub ratio: f64,
    pub threshold: f64,
}

/// Successful gate report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoreVsBoltGateReport {
    pub ratios: Vec<MetricRatio>,
}

/// A concrete reason the perf gate failed.
#[derive(Clone, Debug, PartialEq)]
pub enum PerfGateViolation {
    MissingMetric {
        path: PerfPath,
        metric: PerfMetric,
    },
    InvalidMetric {
        path: PerfPath,
        metric: PerfMetric,
        value: f64,
    },
    InvalidThreshold {
        metric: PerfMetric,
        threshold: f64,
    },
    Regression {
        metric: PerfMetric,
        reference: f64,
        candidate: f64,
        ratio: f64,
        threshold: f64,
    },
    MissingSpan {
        path: PerfPath,
        span: String,
    },
}

/// Check a paired core-vs-Bolt perf gate using the default required span set.
pub fn check_core_vs_bolt_gate(
    core: &PerfMetrics,
    bolt: &PerfMetrics,
    thresholds: PerfGateThresholds,
) -> Result<CoreVsBoltGateReport, Vec<PerfGateViolation>> {
    check_core_vs_bolt_gate_with_spans(core, bolt, thresholds, CORE_VS_BOLT_REQUIRED_SPANS)
}

/// Check a paired core-vs-Bolt perf gate with a caller-provided required span set.
pub fn check_core_vs_bolt_gate_with_spans(
    core: &PerfMetrics,
    bolt: &PerfMetrics,
    thresholds: PerfGateThresholds,
    required_spans: &[&str],
) -> Result<CoreVsBoltGateReport, Vec<PerfGateViolation>> {
    let mut report = CoreVsBoltGateReport::default();
    let mut violations = Vec::new();

    compare_f64_metric(
        &mut report,
        &mut violations,
        PerfMetric::SetupMs,
        core.setup_ms,
        bolt.setup_ms,
        thresholds.max_setup_ratio,
    );
    compare_f64_metric(
        &mut report,
        &mut violations,
        PerfMetric::ProveMs,
        core.prove_ms,
        bolt.prove_ms,
        thresholds.max_prove_ratio,
    );
    compare_f64_metric(
        &mut report,
        &mut violations,
        PerfMetric::VerifyMs,
        core.verify_ms,
        bolt.verify_ms,
        thresholds.max_verify_ratio,
    );
    compare_u64_metric(
        &mut report,
        &mut violations,
        PerfMetric::ProofBytes,
        core.proof_bytes,
        bolt.proof_bytes,
        thresholds.max_proof_size_ratio,
    );
    compare_u64_metric(
        &mut report,
        &mut violations,
        PerfMetric::PeakRssMb,
        core.peak_rss_mb,
        bolt.peak_rss_mb,
        thresholds.max_peak_rss_ratio,
    );

    check_required_spans(&mut violations, core, bolt, required_spans);

    if violations.is_empty() {
        Ok(report)
    } else {
        Err(violations)
    }
}

fn compare_f64_metric(
    report: &mut CoreVsBoltGateReport,
    violations: &mut Vec<PerfGateViolation>,
    metric: PerfMetric,
    reference: Option<f64>,
    candidate: Option<f64>,
    threshold: Option<f64>,
) {
    let Some(threshold) = valid_threshold(violations, metric, threshold) else {
        return;
    };
    let Some(reference) = valid_f64_metric(violations, PerfPath::Core, metric, reference) else {
        return;
    };
    let Some(candidate) = valid_f64_metric(violations, PerfPath::Bolt, metric, candidate) else {
        return;
    };

    compare_ratio(report, violations, metric, reference, candidate, threshold);
}

fn compare_u64_metric(
    report: &mut CoreVsBoltGateReport,
    violations: &mut Vec<PerfGateViolation>,
    metric: PerfMetric,
    reference: Option<u64>,
    candidate: Option<u64>,
    threshold: Option<f64>,
) {
    let Some(threshold) = valid_threshold(violations, metric, threshold) else {
        return;
    };
    let Some(reference) = valid_u64_metric(violations, PerfPath::Core, metric, reference) else {
        return;
    };
    let Some(candidate) = valid_u64_metric(violations, PerfPath::Bolt, metric, candidate) else {
        return;
    };

    compare_ratio(report, violations, metric, reference, candidate, threshold);
}

fn valid_threshold(
    violations: &mut Vec<PerfGateViolation>,
    metric: PerfMetric,
    threshold: Option<f64>,
) -> Option<f64> {
    let threshold = threshold?;
    if threshold.is_finite() && threshold >= 1.0 {
        Some(threshold)
    } else {
        violations.push(PerfGateViolation::InvalidThreshold { metric, threshold });
        None
    }
}

fn valid_f64_metric(
    violations: &mut Vec<PerfGateViolation>,
    path: PerfPath,
    metric: PerfMetric,
    value: Option<f64>,
) -> Option<f64> {
    let Some(value) = value else {
        violations.push(PerfGateViolation::MissingMetric { path, metric });
        return None;
    };
    if value.is_finite() && (path != PerfPath::Core || value > 0.0) {
        Some(value)
    } else {
        violations.push(PerfGateViolation::InvalidMetric {
            path,
            metric,
            value,
        });
        None
    }
}

fn valid_u64_metric(
    violations: &mut Vec<PerfGateViolation>,
    path: PerfPath,
    metric: PerfMetric,
    value: Option<u64>,
) -> Option<f64> {
    let Some(value) = value else {
        violations.push(PerfGateViolation::MissingMetric { path, metric });
        return None;
    };
    if path == PerfPath::Core && value == 0 {
        violations.push(PerfGateViolation::InvalidMetric {
            path,
            metric,
            value: 0.0,
        });
        None
    } else {
        Some(value as f64)
    }
}

fn compare_ratio(
    report: &mut CoreVsBoltGateReport,
    violations: &mut Vec<PerfGateViolation>,
    metric: PerfMetric,
    reference: f64,
    candidate: f64,
    threshold: f64,
) {
    let ratio = candidate / reference;
    report.ratios.push(MetricRatio {
        metric,
        reference,
        candidate,
        ratio,
        threshold,
    });
    if ratio > threshold {
        violations.push(PerfGateViolation::Regression {
            metric,
            reference,
            candidate,
            ratio,
            threshold,
        });
    }
}

fn check_required_spans(
    violations: &mut Vec<PerfGateViolation>,
    core: &PerfMetrics,
    bolt: &PerfMetrics,
    required_spans: &[&str],
) {
    for &span in required_spans {
        let Some(path) = span_path(span) else {
            continue;
        };
        let metrics = match path {
            PerfPath::Core => core,
            PerfPath::Bolt => bolt,
        };
        if !metrics.span_names.iter().any(|observed| observed == span) {
            violations.push(PerfGateViolation::MissingSpan {
                path,
                span: span.to_owned(),
            });
        }
    }
}

fn span_path(span: &str) -> Option<PerfPath> {
    if span.starts_with("core.") {
        Some(PerfPath::Core)
    } else if span.starts_with("bolt.") {
        Some(PerfPath::Bolt)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_core_vs_bolt_gate, check_core_vs_bolt_gate_with_spans, PerfGateThresholds,
        PerfGateViolation, PerfMetric, PerfMetrics, PerfPath, CORE_VS_BOLT_REQUIRED_SPANS,
    };

    fn complete_core_metrics() -> PerfMetrics {
        PerfMetrics {
            setup_ms: Some(10.0),
            prove_ms: Some(100.0),
            verify_ms: Some(20.0),
            proof_bytes: Some(1_000),
            peak_rss_mb: Some(500),
            span_names: CORE_VS_BOLT_REQUIRED_SPANS
                .iter()
                .filter(|span| span.starts_with("core."))
                .map(|span| (*span).to_owned())
                .collect(),
        }
    }

    fn complete_bolt_metrics() -> PerfMetrics {
        PerfMetrics {
            setup_ms: Some(10.0),
            prove_ms: Some(100.0),
            verify_ms: Some(20.0),
            proof_bytes: Some(1_000),
            peak_rss_mb: Some(500),
            span_names: CORE_VS_BOLT_REQUIRED_SPANS
                .iter()
                .filter(|span| span.starts_with("bolt."))
                .map(|span| (*span).to_owned())
                .collect(),
        }
    }

    #[test]
    fn passes_complete_metrics_and_spans() {
        let report = check_core_vs_bolt_gate(
            &complete_core_metrics(),
            &complete_bolt_metrics(),
            PerfGateThresholds::default(),
        );

        assert!(matches!(report, Ok(report) if report.ratios.len() == 4));
    }

    #[test]
    fn reports_regression() {
        let core = complete_core_metrics();
        let mut bolt = complete_bolt_metrics();
        bolt.prove_ms = Some(120.0);

        let result = check_core_vs_bolt_gate(&core, &bolt, PerfGateThresholds::default());

        assert!(
            matches!(
                result,
                Err(violations) if violations.contains(&PerfGateViolation::Regression {
                    metric: PerfMetric::ProveMs,
                    reference: 100.0,
                    candidate: 120.0,
                    ratio: 1.2,
                    threshold: 1.05,
                })
            ),
            "expected prove regression"
        );
    }

    #[test]
    fn reports_missing_required_span() {
        let core = complete_core_metrics();
        let mut bolt = complete_bolt_metrics();
        bolt.span_names.retain(|span| span != "bolt.stage8");

        let result = check_core_vs_bolt_gate(&core, &bolt, PerfGateThresholds::default());

        assert!(
            matches!(
                result,
                Err(violations) if violations.contains(&PerfGateViolation::MissingSpan {
                    path: PerfPath::Bolt,
                    span: "bolt.stage8".to_owned(),
                })
            ),
            "expected missing span violation"
        );
    }

    #[test]
    fn disabled_thresholds_still_check_spans() {
        let result = check_core_vs_bolt_gate_with_spans(
            &PerfMetrics::default().with_span_names(["core.prove"]),
            &PerfMetrics::default(),
            PerfGateThresholds::disabled(),
            &["core.prove", "bolt.prove"],
        );

        assert!(
            matches!(
                result,
                Err(violations) if violations == vec![PerfGateViolation::MissingSpan {
                    path: PerfPath::Bolt,
                    span: "bolt.prove".to_owned(),
                }]
            ),
            "expected only span validation when thresholds are disabled"
        );
    }
}
