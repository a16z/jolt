//! Perf-oracle helpers for equivalence tests.

#![expect(
    clippy::panic,
    clippy::print_stdout,
    reason = "perf oracle gates should fail fast and print successful ratio reports"
)]

use std::collections::BTreeSet;
use std::sync::Once;

use jolt_profiling::{
    check_core_vs_bolt_gate, observed_span_names_with_prefix, setup_tracing, CoreVsBoltGateReport,
    PerfGateThresholds, PerfMetrics, TracingFormat,
};
use serde::Serialize;

static PERF_TRACING: Once = Once::new();

pub const CORE_VS_BOLT_PERF_THRESHOLDS: PerfGateThresholds = PerfGateThresholds {
    max_setup_ratio: None,
    max_prove_ratio: Some(1.3),
    max_verify_ratio: Some(100.0),
    max_proof_size_ratio: Some(100.0),
    max_peak_rss_ratio: Some(100.0),
};

#[derive(Clone, Debug)]
pub struct CoreVsBoltPerfSample {
    pub core: PerfMetrics,
    pub bolt: PerfMetrics,
}

#[derive(Clone, Debug)]
pub struct ProveRatioConfidenceInterval {
    pub sample_count: usize,
    pub confidence: f64,
    pub mean: f64,
    pub stddev: f64,
    pub lower: f64,
    pub upper: f64,
    pub threshold: f64,
}

#[derive(Clone, Debug)]
pub struct SampledCoreVsBoltGateReport {
    pub median_core: PerfMetrics,
    pub median_bolt: PerfMetrics,
    pub median_report: CoreVsBoltGateReport,
    pub prove_ratio_interval: Option<ProveRatioConfidenceInterval>,
}

pub fn core_vs_bolt_perf_sample_count() -> usize {
    match std::env::var("JOLT_BOLT_PERF_SAMPLES") {
        Ok(value) => parse_perf_sample_count(&value),
        Err(_) if std::env::var_os("CI").is_some() => 3,
        Err(_) => 1,
    }
}

pub fn check_sampled_core_vs_bolt_perf_gate(
    samples: &[CoreVsBoltPerfSample],
    thresholds: PerfGateThresholds,
) -> Result<SampledCoreVsBoltGateReport, String> {
    let Some(first) = samples.first() else {
        return Err("core-vs-Bolt perf oracle produced no samples".to_owned());
    };

    if samples.len() == 1 {
        let median_report = check_core_vs_bolt_gate(&first.core, &first.bolt, thresholds)
            .map_err(|violations| format!("core-vs-Bolt perf oracle gate: {violations:?}"))?;
        return Ok(SampledCoreVsBoltGateReport {
            median_core: first.core.clone(),
            median_bolt: first.bolt.clone(),
            median_report,
            prove_ratio_interval: prove_ratio_confidence_interval(samples, thresholds),
        });
    }

    let median_core = median_metrics(samples.iter().map(|sample| &sample.core));
    let median_bolt = median_metrics(samples.iter().map(|sample| &sample.bolt));
    let non_prove_thresholds = PerfGateThresholds {
        max_prove_ratio: None,
        ..thresholds
    };
    let median_report = check_core_vs_bolt_gate(&median_core, &median_bolt, non_prove_thresholds)
        .map_err(|violations| {
        format!("sampled core-vs-Bolt perf oracle non-prove gate: {violations:?}")
    })?;

    let prove_ratio_interval = prove_ratio_confidence_interval(samples, thresholds);
    if let Some(interval) = &prove_ratio_interval {
        if interval.lower > interval.threshold {
            return Err(format!(
                "sampled core-vs-Bolt prove_ms ratio is confidently above threshold: \
                 mean={:.3}x, 95% CI=[{:.3}x, {:.3}x], threshold={:.3}x, samples={}",
                interval.mean,
                interval.lower,
                interval.upper,
                interval.threshold,
                interval.sample_count
            ));
        }
    }

    Ok(SampledCoreVsBoltGateReport {
        median_core,
        median_bolt,
        median_report,
        prove_ratio_interval,
    })
}

/// Installs perf span observation, and enables a Chrome/Perfetto trace when
/// `JOLT_BOLT_PERF_TRACE` is set.
pub fn maybe_setup_perf_trace(trace_name: &'static str) {
    PERF_TRACING.call_once(|| {
        let formats: &[TracingFormat] = if std::env::var_os("JOLT_BOLT_PERF_TRACE").is_some() {
            &[TracingFormat::Chrome]
        } else {
            &[]
        };
        let _ = Box::leak(Box::new(setup_tracing(formats, trace_name)));
    });
}

pub fn generated_bolt_perf_metrics(
    setup_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    proof: &jolt_verifier::JoltProof,
    peak_rss_mb: u64,
) -> PerfMetrics {
    PerfMetrics {
        setup_ms: Some(setup_ms),
        prove_ms: Some(prove_ms),
        verify_ms: Some(verify_ms),
        proof_bytes: Some(generated_jolt_proof_bytes(proof)),
        peak_rss_mb: Some(peak_rss_mb),
        span_names: observed_span_names_with_prefix("bolt."),
    }
}

pub fn print_core_vs_bolt_perf_summary(
    core: &PerfMetrics,
    bolt: &PerfMetrics,
    report: &CoreVsBoltGateReport,
) {
    println!("core-vs-Bolt perf summary:");
    print_f64_metric("setup_ms", core.setup_ms, bolt.setup_ms);
    print_f64_metric("prove_ms", core.prove_ms, bolt.prove_ms);
    print_f64_metric("verify_ms", core.verify_ms, bolt.verify_ms);
    print_u64_metric("proof_bytes", core.proof_bytes, bolt.proof_bytes);
    print_u64_metric("peak_rss_mb", core.peak_rss_mb, bolt.peak_rss_mb);
    for ratio in &report.ratios {
        println!(
            "  gated {}: core={:.3}, bolt={:.3}, ratio={:.3}x, threshold={:.3}x",
            ratio.metric, ratio.reference, ratio.candidate, ratio.ratio, ratio.threshold
        );
    }
}

pub fn print_sampled_core_vs_bolt_perf_summary(
    samples: &[CoreVsBoltPerfSample],
    report: &SampledCoreVsBoltGateReport,
) {
    println!(
        "sampled core-vs-Bolt perf summary: samples={}",
        samples.len()
    );
    print_core_vs_bolt_perf_summary(
        &report.median_core,
        &report.median_bolt,
        &report.median_report,
    );
    if let Some(interval) = &report.prove_ratio_interval {
        println!(
            "  gated prove_ms: mean={:.3}x, stddev={:.3}x, 95% CI=[{:.3}x, {:.3}x], threshold={:.3}x",
            interval.mean, interval.stddev, interval.lower, interval.upper, interval.threshold
        );
    }
}

fn parse_perf_sample_count(value: &str) -> usize {
    let count = value
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("parse JOLT_BOLT_PERF_SAMPLES={value:?}: {error}"));
    assert!(
        count > 0,
        "JOLT_BOLT_PERF_SAMPLES must be greater than zero"
    );
    count
}

fn median_metrics<'a>(metrics: impl Iterator<Item = &'a PerfMetrics>) -> PerfMetrics {
    let metrics: Vec<_> = metrics.collect();
    PerfMetrics {
        setup_ms: median_f64(metrics.iter().filter_map(|metrics| metrics.setup_ms)),
        prove_ms: median_f64(metrics.iter().filter_map(|metrics| metrics.prove_ms)),
        verify_ms: median_f64(metrics.iter().filter_map(|metrics| metrics.verify_ms)),
        proof_bytes: median_u64(metrics.iter().filter_map(|metrics| metrics.proof_bytes)),
        peak_rss_mb: median_u64(metrics.iter().filter_map(|metrics| metrics.peak_rss_mb)),
        span_names: union_span_names(metrics.iter().map(|metrics| metrics.span_names.as_slice())),
    }
}

fn median_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<_> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len() % 2 == 0 {
        Some(f64::midpoint(values[midpoint - 1], values[midpoint]))
    } else {
        Some(values[midpoint])
    }
}

fn median_u64(values: impl Iterator<Item = u64>) -> Option<u64> {
    let mut values: Vec<_> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn union_span_names<'a>(span_names: impl Iterator<Item = &'a [String]>) -> Vec<String> {
    let mut names = BTreeSet::new();
    for sample_names in span_names {
        names.extend(sample_names.iter().cloned());
    }
    names.into_iter().collect()
}

fn prove_ratio_confidence_interval(
    samples: &[CoreVsBoltPerfSample],
    thresholds: PerfGateThresholds,
) -> Option<ProveRatioConfidenceInterval> {
    let threshold = thresholds.max_prove_ratio?;
    let ratios: Vec<_> = samples
        .iter()
        .filter_map(|sample| {
            let core = sample.core.prove_ms?;
            let bolt = sample.bolt.prove_ms?;
            (core > 0.0).then_some(bolt / core)
        })
        .collect();
    let sample_count = ratios.len();
    if sample_count == 0 {
        return None;
    }
    let mean = ratios.iter().sum::<f64>() / sample_count as f64;
    let stddev = if sample_count > 1 {
        let variance = ratios
            .iter()
            .map(|ratio| {
                let delta = ratio - mean;
                delta * delta
            })
            .sum::<f64>()
            / (sample_count - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let half_width = if sample_count > 1 {
        student_t_95_two_sided(sample_count - 1) * stddev / (sample_count as f64).sqrt()
    } else {
        0.0
    };
    Some(ProveRatioConfidenceInterval {
        sample_count,
        confidence: 0.95,
        mean,
        stddev,
        lower: mean - half_width,
        upper: mean + half_width,
        threshold,
    })
}

fn student_t_95_two_sided(degrees_of_freedom: usize) -> f64 {
    match degrees_of_freedom {
        0 => 0.0,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11..=20 => 2.086,
        21..=30 => 2.042,
        _ => 1.96,
    }
}

fn print_f64_metric(label: &str, core: Option<f64>, bolt: Option<f64>) {
    match (core, bolt) {
        (Some(core), Some(bolt)) if core > 0.0 => {
            println!(
                "  {label}: core={core:.3}, bolt={bolt:.3}, ratio={:.3}x",
                bolt / core
            );
        }
        (Some(core), Some(bolt)) => {
            println!("  {label}: core={core:.3}, bolt={bolt:.3}, ratio=n/a");
        }
        _ => println!("  {label}: unavailable"),
    }
}

fn print_u64_metric(label: &str, core: Option<u64>, bolt: Option<u64>) {
    match (core, bolt) {
        (Some(core), Some(bolt)) if core > 0 => {
            println!(
                "  {label}: core={core}, bolt={bolt}, ratio={:.3}x",
                bolt as f64 / core as f64
            );
        }
        (Some(core), Some(bolt)) => {
            println!("  {label}: core={core}, bolt={bolt}, ratio=n/a");
        }
        _ => println!("  {label}: unavailable"),
    }
}

fn generated_jolt_proof_bytes(proof: &jolt_verifier::JoltProof) -> u64 {
    [
        serialized_component_size(&proof.commitments, "jolt proof commitments"),
        serialized_component_size(&proof.stage1_outer, "jolt proof stage1"),
        serialized_component_size(&proof.stage2, "jolt proof stage2"),
        serialized_component_size(&proof.stage3, "jolt proof stage3"),
        serialized_component_size(&proof.stage4, "jolt proof stage4"),
        serialized_component_size(&proof.stage5, "jolt proof stage5"),
        serialized_component_size(&proof.stage6, "jolt proof stage6"),
        serialized_component_size(&proof.stage7, "jolt proof stage7"),
        proof.evaluation.as_ref().map_or(0, |evaluation| {
            serialized_component_size(&evaluation.joint_opening_proof, "jolt proof evaluation")
        }),
    ]
    .into_iter()
    .sum()
}

fn serialized_component_size<T: Serialize>(component: &T, label: &str) -> u64 {
    postcard::to_stdvec(component)
        .unwrap_or_else(|error| panic!("serialize {label}: {error}"))
        .len() as u64
}
