//! E2e smoke test for the profile harness: one in-process fibonacci run
//! must emit both telemetry artifacts, the summary must parse through the
//! strict schema structs, and every taxonomy-v1 label that fires on all
//! proves must be present in the trace — so a silent span rename fails CI
//! rather than drifting.
//!
//! Scale 2^13 — fibonacci's minimum guest scale. Label coverage is
//! scale-independent. Compiled with the `akita` feature the same run drives
//! the packed prover and asserts its presence set (and the `_akita`-suffixed
//! artifact names).
//!
//! NOT wired into CI yet: the reference backend's naive RAM kernels retain
//! ~18 GiB regardless of trace length (`ram_K` is priced off the guest's
//! default 32 MB heap, not the trace), which exceeds hosted-runner memory.
//! Hook up a dedicated `rust.yml` job (guest toolchain + jolt CLI, like the
//! legacy test jobs) once an optimized backend fits runner memory.
//!
//! Run explicitly (needs the guest toolchain, like the byte-diff harness):
//! `cargo nextest run -p jolt-prover --features profiling -E 'binary(profiling_smoke)'`

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
        scale: Some(13),
        format: OutputFormat::Chrome,
        backend: BackendKind::Reference,
    });

    let trace_path = artifacts.trace_path.expect("trace path");
    let summary_path = artifacts.summary_path.expect("summary path");
    // Artifacts are grouped into a per-run directory
    // (benchmark-runs/{timestamp}_modular_fibonacci_13/, suffixed `_akita`
    // on the packed build), with the `latest_` link pointing at this run;
    // the directory name carries the run identity, so the files inside use
    // fixed names.
    let stem = if cfg!(feature = "akita") {
        "modular_fibonacci_akita_13"
    } else {
        "modular_fibonacci_13"
    };
    assert_eq!(trace_path.file_name().unwrap(), "trace.json");
    assert_eq!(summary_path.file_name().unwrap(), "summary.json");
    assert_eq!(summary_path.parent(), trace_path.parent());
    let run_dir = trace_path.parent().unwrap();
    let dir_name = run_dir.file_name().unwrap().to_str().unwrap();
    let (timestamp, rest) = dir_name.split_at(15);
    assert_eq!(rest, format!("_{stem}"), "run dir: {dir_name}");
    assert!(
        timestamp.chars().enumerate().all(|(i, c)| if i == 8 {
            c == '-'
        } else {
            c.is_ascii_digit()
        }),
        "timestamp prefix: {timestamp}"
    );
    assert_eq!(
        std::fs::canonicalize(format!("benchmark-runs/latest_{stem}")).unwrap(),
        std::fs::canonicalize(run_dir).unwrap(),
        "latest link resolves to this run"
    );

    // Both artifacts exist and parse; the summary parses through the strict
    // (`deny_unknown_fields`) schema structs — the instance-level validation
    // against the checked-in JSON Schema, which a fixture test keeps in sync
    // with those structs.
    let trace: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    let summary: ProfileSummary =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).unwrap()).unwrap();

    // Every always-present taxonomy-v1 label fired, for the mode this
    // prover was compiled in — the `zk` feature swaps the uni-skip and
    // stage-8 opening seams for their committed siblings, and the `akita`
    // feature swaps the commitment seams for the packed set. (The advice
    // seams are exempt: fibonacci exercises no advice.)
    let mode = if cfg!(feature = "akita") {
        taxonomy::ProverMode::Akita
    } else if cfg!(feature = "zk") {
        taxonomy::ProverMode::Zk
    } else {
        taxonomy::ProverMode::Clear
    };
    let emitted: std::collections::HashSet<&str> = trace
        .iter()
        .filter(|e| e.get("ph").and_then(Value::as_str) == Some("B"))
        .filter_map(|e| e.get("name").and_then(Value::as_str))
        .collect();
    let missing: Vec<&str> = taxonomy::always_present_spans(mode)
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
    assert_eq!(summary.run.scale_log2, 13);
    assert!(summary.peak_rss_gib.is_some());

    // The counter rewrite ran: no raw `counters.*` events survive in the
    // trace, and the monitor's samples aggregated into the summary.
    assert!(trace.iter().all(|e| {
        e.get("args")
            .and_then(Value::as_object)
            .is_none_or(|args| !args.keys().any(|k| k.starts_with("counters.")))
    }));
}
