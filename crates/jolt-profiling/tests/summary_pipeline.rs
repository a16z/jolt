//! Fixture-trace tests for the flush-time summary pipeline: counter
//! conversion, aggregation semantics (self time, dark time, stage windows),
//! trace/summary consistency, and schema drift.

#![cfg(all(not(target_arch = "wasm32"), feature = "summary"))]
#![expect(clippy::unwrap_used)]

use jolt_profiling::stage_memory::StageMemoryRow;
use jolt_profiling::summary::{
    build_summary, convert_counter_events, parse_folded, summary_path, ProfileSummary,
    SummaryContext, SUMMARY_SCHEMA_JSON,
};
use jolt_profiling::taxonomy;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/simple_trace.json");
const GIB: f64 = 1_073_741_824.0;

fn fixture_events() -> Vec<Value> {
    serde_json::from_str(FIXTURE).unwrap()
}

fn fixture_context() -> SummaryContext {
    SummaryContext {
        workload: "fibonacci".to_string(),
        scale_log2: 16,
        backend: "reference".to_string(),
    }
}

fn fixture_stage_rows() -> Vec<StageMemoryRow> {
    vec![StageMemoryRow {
        stage: "prove_stage0",
        rss_open_bytes: GIB as u64,
        rss_close_bytes: 2 * GIB as u64,
    }]
}

fn fixture_summary(events: &[Value]) -> ProfileSummary {
    build_summary(
        events,
        &fixture_context(),
        &fixture_stage_rows(),
        Some(4 * GIB as u64),
        1_700_000_000,
        Some("abc1234".to_string()),
        // Exercise the heap section through the same strict-schema tests:
        // one snapshot parsed from a folded-stacks blob, as the allocative
        // lane would supply.
        [(
            "Stage2Batch_prepared".to_string(),
            parse_folded("KernelA;opening_tables 6442450944\nKernelA;derived_tables 2147483648\nProofSession 1024\n"),
        )]
        .into_iter()
        .collect(),
    )
}

#[test]
fn counter_events_convert_to_chrome_counter_tracks() {
    let events = fixture_events();
    let original_len = events.len();
    let converted = convert_counter_events(events);

    // 3 monitor instants carrying 4 counter samples total → 4 "C" events.
    let counters: Vec<&Value> = converted
        .iter()
        .filter(|e| e.get("ph").and_then(Value::as_str) == Some("C"))
        .collect();
    assert_eq!(counters.len(), 4);
    assert_eq!(converted.len(), original_len - 3 + 4);
    // No raw counter instants survive; non-counter instants (the
    // `heap_snapshot` marker) pass through untouched.
    assert!(converted.iter().all(|e| {
        e.get("args")
            .and_then(Value::as_object)
            .is_none_or(|args| !args.keys().any(|k| k.starts_with("counters.")))
    }));
    assert_eq!(
        converted
            .iter()
            .filter(|e| e.get("ph").and_then(Value::as_str) == Some("i"))
            .count(),
        1,
        "the heap_snapshot instant survives conversion"
    );

    let memory: Vec<&Value> = counters
        .iter()
        .filter(|e| e.get("name").and_then(Value::as_str) == Some("memory_gib"))
        .copied()
        .collect();
    assert_eq!(memory.len(), 3);
    assert_eq!(
        memory[0].get("args").and_then(|a| a.get("memory_gib")),
        Some(&Value::from(1.0))
    );
    // Placement metadata survives the rewrite.
    assert_eq!(memory[0].get("ts"), Some(&Value::from(500.0)));
    assert_eq!(memory[0].get("tid"), Some(&Value::from(3)));
}

#[test]
fn aggregation_computes_totals_self_and_dark_time() {
    let summary = fixture_summary(&fixture_events());

    let span = |label: &str| summary.spans.get(label).unwrap();
    // Inclusive totals.
    assert_eq!(span("jolt_prover::prove").total_ns, 2_500_000);
    assert_eq!(span("prove_stage0").total_ns, 900_000);
    assert_eq!(span("commit_witness").total_ns, 500_000);
    assert_eq!(span("prove_stage1").total_ns, 1_000_000);
    assert_eq!(span("Stage1Batch::prove").total_ns, 700_000);
    assert_eq!(span("prove_batch").total_ns, 500_000);
    assert_eq!(span("sumcheck_round").count, 2);
    assert_eq!(span("sumcheck_round").total_ns, 300_000);
    // Self time subtracts same-thread children only: the tid-2
    // EqPolynomial::evals span never subtracts from the tid-1 stack.
    assert_eq!(span("prove_batch").self_ns, 200_000);
    assert_eq!(span("Stage1Batch::prove").self_ns, 200_000);
    assert_eq!(span("prove_stage1").self_ns, 300_000);
    assert_eq!(span("EqPolynomial::evals").total_ns, 100_000);
    assert_eq!(span("EqPolynomial::evals").self_ns, 100_000);

    // Dark time at the root: 2500µs wall − (900 + 1000)µs stage children.
    let root = summary.root.as_ref().unwrap();
    assert_eq!(root.label, taxonomy::ROOT_SPAN);
    assert_eq!(root.wall_time_ns, 2_500_000);
    assert_eq!(root.dark_time_ns, 600_000);
    assert!((root.dark_time_fraction - 0.24).abs() < 1e-9);
    assert_eq!(root.peak_memory_gib, Some(3.0));
}

#[test]
fn stage_rollup_folds_boundary_rss_and_windowed_peaks() {
    let summary = fixture_summary(&fixture_events());

    assert_eq!(summary.stages.len(), 2);
    let stage0 = &summary.stages[0];
    assert_eq!(stage0.label, "prove_stage0");
    assert_eq!(stage0.wall_time_ns, 900_000);
    // Boundary RSS from the StageMemoryLayer row: 1 GiB → 2 GiB.
    assert_eq!(stage0.rss_open_gib, Some(1.0));
    assert_eq!(stage0.rss_close_gib, Some(2.0));
    assert_eq!(stage0.rss_delta_gib, Some(1.0));
    // Only the ts=500 sample falls inside [100, 1000].
    assert_eq!(stage0.peak_memory_gib, Some(1.0));

    let stage1 = &summary.stages[1];
    assert_eq!(stage1.label, "prove_stage1");
    assert_eq!(stage1.wall_time_ns, 1_000_000);
    // No StageMemoryRow for stage 1 → nullable boundary fields.
    assert_eq!(stage1.rss_open_gib, None);
    assert_eq!(stage1.rss_delta_gib, None);
    assert_eq!(stage1.peak_memory_gib, Some(3.0));

    assert_eq!(summary.peak_rss_gib, Some(4.0));
    let memory = summary.counters.get("memory_gib").unwrap();
    assert_eq!(memory.samples, 3);
    assert_eq!(memory.max, 3.0);
    assert!((memory.mean - 2.0).abs() < 1e-9);
    let cpu = summary.counters.get("cpu_percent").unwrap();
    assert_eq!(cpu.samples, 1);
}

/// Trace/summary consistency: the summary is a deterministic aggregation of
/// the trace's events, and the counter rewrite does not change it.
#[test]
fn summary_is_invariant_under_counter_conversion() {
    let raw = fixture_summary(&fixture_events());
    let converted = fixture_summary(&convert_counter_events(fixture_events()));
    assert_eq!(
        serde_json::to_value(&raw).unwrap(),
        serde_json::to_value(&converted).unwrap()
    );
}

/// Taxonomy conformance over the fixture: stage labels are taxonomy members,
/// the root span carries its required field, and the always-present label
/// set is internally consistent.
#[test]
fn fixture_labels_conform_to_taxonomy() {
    let events = fixture_events();
    for event in &events {
        let name = event
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if name.starts_with("prove_stage") {
            assert!(
                taxonomy::STAGE_SPANS.contains(&name),
                "unknown stage label {name}"
            );
        }
        if name == taxonomy::ROOT_SPAN && event.get("ph").and_then(Value::as_str) == Some("B") {
            assert!(
                event
                    .get("args")
                    .and_then(|a| a.get("trace_length"))
                    .is_some(),
                "root span must carry trace_length"
            );
        }
    }
    for mode in [taxonomy::ProverMode::Clear, taxonomy::ProverMode::Zk] {
        let always = taxonomy::always_present_spans(mode);
        assert!(always.contains(&taxonomy::ROOT_SPAN));
        assert!(always.contains(&"prove_batch"));
        assert!(!always
            .iter()
            .any(|l| taxonomy::ADVICE_SEAM_SPANS.contains(l)));
    }
    // The mode seams are disjoint siblings: exactly one pair per mode.
    let clear = taxonomy::always_present_spans(taxonomy::ProverMode::Clear);
    let zk = taxonomy::always_present_spans(taxonomy::ProverMode::Zk);
    assert!(taxonomy::CLEAR_MODE_SPANS
        .iter()
        .all(|l| clear.contains(l) && !zk.contains(l)));
    assert!(taxonomy::ZK_MODE_SPANS
        .iter()
        .all(|l| zk.contains(l) && !clear.contains(l)));
}

/// Repeated stage labels pair with their rows by occurrence index (both are
/// recorded in span-close order), never all with the first matching row.
#[test]
fn repeated_stage_labels_pair_rows_by_occurrence() {
    let events: Vec<Value> = serde_json::from_str(
        r#"[
        {"ph":"B","name":"prove_stage0","ts":100.0,"pid":1,"tid":1},
        {"ph":"E","name":"prove_stage0","ts":200.0,"pid":1,"tid":1},
        {"ph":"B","name":"prove_stage0","ts":300.0,"pid":1,"tid":1},
        {"ph":"E","name":"prove_stage0","ts":700.0,"pid":1,"tid":1}
    ]"#,
    )
    .unwrap();
    let rows = vec![
        StageMemoryRow {
            stage: "prove_stage0",
            rss_open_bytes: GIB as u64,
            rss_close_bytes: 2 * GIB as u64,
        },
        StageMemoryRow {
            stage: "prove_stage0",
            rss_open_bytes: 3 * GIB as u64,
            rss_close_bytes: 5 * GIB as u64,
        },
    ];
    let summary = build_summary(
        &events,
        &fixture_context(),
        &rows,
        None,
        0,
        None,
        Default::default(),
    );

    assert_eq!(summary.stages.len(), 2);
    // First close (100→200µs) gets row 0, second (300→700µs) row 1.
    assert_eq!(summary.stages[0].wall_time_ns, 100_000);
    assert_eq!(summary.stages[0].rss_open_gib, Some(1.0));
    assert_eq!(summary.stages[0].rss_close_gib, Some(2.0));
    assert_eq!(summary.stages[1].wall_time_ns, 400_000);
    assert_eq!(summary.stages[1].rss_open_gib, Some(3.0));
    assert_eq!(summary.stages[1].rss_delta_gib, Some(2.0));
}

/// The flush-time I/O wrapper: counter events rewritten in the trace (via
/// temp file + rename — no `.tmp` residue), the caller-sampled peak RSS
/// carried into the summary, both artifacts parseable afterwards.
#[test]
fn finalize_trace_rewrites_and_summarizes_atomically() {
    let dir = std::env::temp_dir().join(format!(
        "jolt_profiling_finalize_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let trace_path = dir.join("trace.json");
    std::fs::write(&trace_path, FIXTURE).unwrap();

    let (out_path, summary) = jolt_profiling::summary::finalize_trace(
        &trace_path,
        &fixture_context(),
        Some(4 * GIB as u64),
    )
    .unwrap();

    assert_eq!(out_path, summary_path(&trace_path));
    assert_eq!(summary.peak_rss_gib, Some(4.0));
    // The rewrite converted every counter instant into a "C" event
    // (non-counter instants like `heap_snapshot` pass through).
    let rewritten: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    assert!(rewritten.iter().all(|e| {
        e.get("args")
            .and_then(Value::as_object)
            .is_none_or(|args| !args.keys().any(|k| k.starts_with("counters.")))
    }));
    // Atomic replacement leaves no temp files behind.
    assert!(std::fs::read_dir(&dir).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    // The written summary parses through the strict schema structs.
    let reparsed: ProfileSummary =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(reparsed.peak_rss_gib, Some(4.0));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Deserializing the summary through the strict serde structs is the
/// instance-level schema validation used by the smoke test.
#[test]
fn summary_round_trips_through_strict_schema_structs() {
    let summary = fixture_summary(&fixture_events());
    let json = serde_json::to_string(&summary).unwrap();
    let reparsed: ProfileSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(reparsed.schema_version, summary.schema_version);
    assert_eq!(reparsed.spans.len(), summary.spans.len());
}

/// The driver's `heap_snapshot` instant events situate each snapshot on the
/// trace clock: the fixture fires one for `Stage2Batch_prepared` at
/// 1350 µs, and the root opens at 0 — so the joined `at_ns` is 1.35 ms.
#[test]
fn heap_snapshots_join_their_instant_events() {
    let summary = fixture_summary(&fixture_events());
    assert_eq!(
        summary.heap["Stage2Batch_prepared"].at_ns,
        Some(1_350_000),
        "snapshot instant joins by label, ns since the root span opened"
    );
}

/// Drift lock between the serde structs (normative) and the checked-in JSON
/// Schema. Regenerate with:
/// `JOLT_UPDATE_SUMMARY_SCHEMA=1 cargo nextest run -p jolt-profiling schema`
#[test]
fn checked_in_schema_matches_structs() {
    let generated = serde_json::to_string_pretty(&schemars::schema_for!(ProfileSummary)).unwrap();
    if std::env::var("JOLT_UPDATE_SUMMARY_SCHEMA").is_ok() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/schema/summary.schema.json");
        std::fs::write(path, format!("{generated}\n")).unwrap();
    }
    assert_eq!(
        SUMMARY_SCHEMA_JSON.trim(),
        generated.trim(),
        "schema drift: run JOLT_UPDATE_SUMMARY_SCHEMA=1 cargo nextest run -p jolt-profiling schema"
    );
}

#[test]
fn summary_path_derives_from_trace_path() {
    assert_eq!(
        summary_path(std::path::Path::new(
            "benchmark-runs/20260801-000000_modular_fibonacci_16/trace.json"
        )),
        std::path::Path::new("benchmark-runs/20260801-000000_modular_fibonacci_16/summary.json")
    );
}

#[test]
fn folded_stacks_parse_into_per_root_totals() {
    let snapshot = parse_folded(
        "KernelA;opening_tables 100\nKernelA;derived_tables 50\nProofSession 8\nnot a folded line\n",
    );
    assert_eq!(snapshot.total_bytes, 158);
    assert_eq!(snapshot.roots["KernelA"], 150);
    assert_eq!(snapshot.roots["ProofSession"], 8);
    assert_eq!(snapshot.roots.len(), 2);
}
