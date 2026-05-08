//! Perf-oracle helpers for equivalence tests.

#![expect(
    clippy::panic,
    clippy::print_stdout,
    reason = "perf oracle gates should fail fast and print successful ratio reports"
)]

use std::sync::Once;

use jolt_profiling::{
    observed_span_names_with_prefix, setup_tracing, CoreVsBoltGateReport, PerfGateThresholds,
    PerfMetrics, TracingFormat,
};
use serde::Serialize;

static PERF_TRACING: Once = Once::new();

pub const CORE_VS_BOLT_PERF_THRESHOLDS: PerfGateThresholds = PerfGateThresholds {
    max_setup_ratio: None,
    max_prove_ratio: Some(100.0),
    max_verify_ratio: Some(100.0),
    max_proof_size_ratio: Some(100.0),
    max_peak_rss_ratio: Some(100.0),
};

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
