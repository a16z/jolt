//! E2e smoke test for the profile harness: one in-process fibonacci run at
//! scale 2^16 must emit both telemetry artifacts, the summary must parse
//! through the strict schema structs, and every taxonomy-v1 label that fires
//! on all proves must be present in the trace — so a silent span rename
//! fails CI rather than drifting.
//!
//! Run explicitly (needs the guest toolchain, like the byte-diff harness):
//! `cargo nextest run -p jolt-prover --features profiling profiling_smoke`

#![cfg(feature = "profiling")]
#![expect(clippy::unwrap_used, clippy::expect_used)]

use jolt_profiling::summary::ProfileSummary;
use jolt_profiling::taxonomy;
use jolt_prover::profile::{BackendKind, OutputFormat, ProfileArgs, Workload};
use serde_json::Value;

#[test]
fn profile_run_emits_conformant_artifacts() {
    let artifacts = jolt_prover::profile::run(&ProfileArgs {
        name: Workload::Fibonacci,
        scale: Some(16),
        format: OutputFormat::Chrome,
        backend: BackendKind::Reference,
    });

    let trace_path = artifacts.trace_path.expect("trace path");
    let summary_path = artifacts.summary_path.expect("summary path");
    assert!(trace_path.ends_with("benchmark-runs/perfetto_traces/modular_fibonacci_16.json"));
    assert!(
        summary_path.ends_with("benchmark-runs/perfetto_traces/modular_fibonacci_16.summary.json")
    );

    // Both artifacts exist and parse; the summary parses through the strict
    // (`deny_unknown_fields`) schema structs — the instance-level validation
    // against the checked-in JSON Schema, which a fixture test keeps in sync
    // with those structs.
    let trace: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    let summary: ProfileSummary =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).unwrap()).unwrap();

    // Every always-present taxonomy-v1 label fired. (`commit_advice` and
    // `AdviceOpeningEvaluation::evaluate` are exempt: fibonacci exercises no
    // advice.)
    let emitted: std::collections::HashSet<&str> = trace
        .iter()
        .filter(|e| e.get("ph").and_then(Value::as_str) == Some("B"))
        .filter_map(|e| e.get("name").and_then(Value::as_str))
        .collect();
    let missing: Vec<&str> = taxonomy::always_present_spans()
        .into_iter()
        .filter(|label| !emitted.contains(label))
        .collect();
    assert!(
        missing.is_empty(),
        "missing taxonomy-v1 labels: {missing:?}"
    );

    // Headline summary sanity: root present with a positive wall time and
    // every stage rolled up with boundary RSS from the StageMemoryLayer.
    let root = summary.root.expect("root summary");
    assert_eq!(root.label, taxonomy::ROOT_SPAN);
    assert!(root.wall_time_ns > 0);
    assert!(root.dark_time_fraction >= 0.0 && root.dark_time_fraction <= 1.0);
    assert_eq!(summary.stages.len(), taxonomy::STAGE_SPANS.len());
    assert!(summary.stages.iter().all(|s| s.rss_open_gib.is_some()));
    assert_eq!(summary.run.workload, "fibonacci");
    assert_eq!(summary.run.scale_log2, 16);
    assert!(summary.peak_rss_gib.is_some());

    // The counter rewrite ran: no raw `counters.*` events survive in the
    // trace, and the monitor's samples aggregated into the summary.
    assert!(trace.iter().all(|e| {
        e.get("args")
            .and_then(Value::as_object)
            .is_none_or(|args| !args.keys().any(|k| k.starts_with("counters.")))
    }));
}
