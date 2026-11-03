#![cfg(all(not(target_arch = "wasm32"), feature = "summary"))]
#![expect(clippy::unwrap_used)]

use jolt_profiling::span_boundary::{BoundarySamples, SpanInstructions, StageBoundaryRow};
use jolt_profiling::summary::{
    build_summary, convert_counter_events, parse_folded, summary_path, ProfileSummary,
    SummaryContext, SUMMARY_SCHEMA_JSON,
};
use jolt_profiling::taxonomy::{self, ROOT_SPAN};
use jolt_profiling::PeakMemory;
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

fn fixture_samples() -> BoundarySamples {
    let sampled = |instances, instructions| SpanInstructions {
        instances,
        instructions,
    };
    BoundarySamples {
        stage_rows: vec![
            StageBoundaryRow {
                stage: "prove_stage0",
                rss_open_bytes: GIB as u64,
                rss_close_bytes: 2 * GIB as u64,
                instructions: Some(7_000_000),
            },
            StageBoundaryRow {
                stage: ROOT_SPAN,
                rss_open_bytes: GIB as u64,
                rss_close_bytes: 3 * GIB as u64,
                instructions: Some(20_000_000),
            },
        ],
        // `EqPolynomial::evals` ran on a pool thread, so it was never
        // sampled.
        span_instructions: [
            (ROOT_SPAN, sampled(1, Some(20_000_000))),
            ("prove_batch", sampled(1, Some(500_000))),
            ("sumcheck_round", sampled(1, Some(200_000))),
            ("commit_witness", sampled(1, None)),
        ]
        .into_iter()
        .collect(),
    }
}

fn fixture_peak() -> PeakMemory {
    PeakMemory {
        rss_bytes: Some(4 * GIB as u64),
        footprint_bytes: Some(6 * GIB as u64),
    }
}

fn fixture_summary(events: &[Value]) -> ProfileSummary {
    build_summary(
        events,
        &fixture_context(),
        &fixture_samples(),
        fixture_peak(),
        1_700_000_000,
        Some("abc1234".to_string()),
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

    let counters: Vec<&Value> = converted
        .iter()
        .filter(|e| e.get("ph").and_then(Value::as_str) == Some("C"))
        .collect();
    assert_eq!(counters.len(), 4);
    assert_eq!(converted.len(), original_len - 3 + 4);
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
    assert_eq!(memory[0].get("ts"), Some(&Value::from(500.0)));
    assert_eq!(memory[0].get("tid"), Some(&Value::from(3)));
}

#[test]
fn aggregation_computes_totals_self_and_dark_time() {
    let summary = fixture_summary(&fixture_events());

    let span = |label: &str| summary.spans.get(label).unwrap();
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

    let root = summary.root.as_ref().unwrap();
    assert_eq!(root.label, ROOT_SPAN);
    assert_eq!(root.wall_time_ns, 2_500_000);
    assert_eq!(root.dark_time_ns, 600_000);
    assert!((root.dark_time_fraction - 0.24).abs() < 1e-9);
    assert_eq!(root.peak_memory_gib, Some(3.0));
    assert_eq!(root.instructions, Some(20_000_000));
}

#[test]
fn stage_rollup_folds_boundary_rss_and_windowed_peaks() {
    let summary = fixture_summary(&fixture_events());

    assert_eq!(summary.stages.len(), 2);
    let stage0 = &summary.stages[0];
    assert_eq!(stage0.label, "prove_stage0");
    assert_eq!(stage0.wall_time_ns, 900_000);
    assert_eq!(stage0.rss_open_gib, Some(1.0));
    assert_eq!(stage0.rss_close_gib, Some(2.0));
    assert_eq!(stage0.rss_delta_gib, Some(1.0));
    assert_eq!(stage0.peak_memory_gib, Some(1.0));
    assert_eq!(stage0.instructions, Some(7_000_000));

    let stage1 = &summary.stages[1];
    assert_eq!(stage1.label, "prove_stage1");
    assert_eq!(stage1.wall_time_ns, 1_000_000);
    assert_eq!(stage1.rss_open_gib, None);
    assert_eq!(stage1.rss_delta_gib, None);
    assert_eq!(stage1.peak_memory_gib, Some(3.0));
    assert_eq!(stage1.instructions, None);

    assert_eq!(summary.peak_rss_gib, Some(4.0));
    assert_eq!(summary.peak_footprint_gib, Some(6.0));
    let memory = summary.counters.get("memory_gib").unwrap();
    assert_eq!(memory.samples, 3);
    assert_eq!(memory.max, 3.0);
    assert!((memory.mean - 2.0).abs() < 1e-9);
    let cpu = summary.counters.get("cpu_percent").unwrap();
    assert_eq!(cpu.samples, 1);
}

/// A label's instruction total is reported only when every trace instance
/// was sampled with a count; partial or unavailable samples are `null`,
/// never an undercount.
#[test]
fn span_instructions_require_every_instance_sampled() {
    let summary = fixture_summary(&fixture_events());
    let instructions = |label: &str| summary.spans.get(label).unwrap().instructions;
    assert_eq!(instructions(ROOT_SPAN), Some(20_000_000));
    assert_eq!(instructions("prove_batch"), Some(500_000));
    assert_eq!(
        instructions("sumcheck_round"),
        None,
        "1 of 2 instances sampled"
    );
    assert_eq!(instructions("commit_witness"), None, "counter unavailable");
    assert_eq!(instructions("EqPolynomial::evals"), None, "pool thread");
    assert_eq!(instructions("prove_stage1"), None, "never sampled");
}

#[test]
fn summary_is_invariant_under_counter_conversion() {
    let raw = fixture_summary(&fixture_events());
    let converted = fixture_summary(&convert_counter_events(fixture_events()));
    assert_eq!(
        serde_json::to_value(&raw).unwrap(),
        serde_json::to_value(&converted).unwrap()
    );
}

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
        if name == ROOT_SPAN && event.get("ph").and_then(Value::as_str) == Some("B") {
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
        assert!(always.contains(&ROOT_SPAN));
        assert!(always.contains(&"prove_batch"));
        assert!(!always
            .iter()
            .any(|l| taxonomy::ADVICE_SEAM_SPANS.contains(l)));
    }
    let clear = taxonomy::always_present_spans(taxonomy::ProverMode::Clear);
    let zk = taxonomy::always_present_spans(taxonomy::ProverMode::Zk);
    assert!(taxonomy::CLEAR_MODE_SPANS
        .iter()
        .all(|l| clear.contains(l) && !zk.contains(l)));
    assert!(taxonomy::ZK_MODE_SPANS
        .iter()
        .all(|l| zk.contains(l) && !clear.contains(l)));
}

/// Repeated stage and root labels pair with their rows by occurrence index
/// (both are recorded in span-exit order), never all with the first
/// matching row; the summary's root is the longest instance.
#[test]
fn repeated_labels_pair_rows_by_occurrence() {
    let events: Vec<Value> = serde_json::from_str(
        r#"[
        {"ph":"B","name":"jolt_prover::prove","ts":0.0,"pid":1,"tid":1},
        {"ph":"B","name":"prove_stage0","ts":100.0,"pid":1,"tid":1},
        {"ph":"E","name":"prove_stage0","ts":200.0,"pid":1,"tid":1},
        {"ph":"E","name":"jolt_prover::prove","ts":250.0,"pid":1,"tid":1},
        {"ph":"B","name":"jolt_prover::prove","ts":250.0,"pid":1,"tid":1},
        {"ph":"B","name":"prove_stage0","ts":300.0,"pid":1,"tid":1},
        {"ph":"E","name":"prove_stage0","ts":700.0,"pid":1,"tid":1},
        {"ph":"E","name":"jolt_prover::prove","ts":800.0,"pid":1,"tid":1}
    ]"#,
    )
    .unwrap();
    let row = |stage, rss_open_gib: u64, rss_close_gib: u64, instructions| StageBoundaryRow {
        stage,
        rss_open_bytes: rss_open_gib * GIB as u64,
        rss_close_bytes: rss_close_gib * GIB as u64,
        instructions: Some(instructions),
    };
    let samples = BoundarySamples {
        stage_rows: vec![
            row("prove_stage0", 1, 2, 10),
            row(ROOT_SPAN, 1, 2, 30),
            row("prove_stage0", 3, 5, 40),
            row(ROOT_SPAN, 2, 5, 60),
        ],
        span_instructions: Default::default(),
    };
    let summary = build_summary(
        &events,
        &fixture_context(),
        &samples,
        PeakMemory::default(),
        0,
        None,
        Default::default(),
    );

    assert_eq!(summary.stages.len(), 2);
    assert_eq!(summary.stages[0].wall_time_ns, 100_000);
    assert_eq!(summary.stages[0].rss_open_gib, Some(1.0));
    assert_eq!(summary.stages[0].rss_close_gib, Some(2.0));
    assert_eq!(summary.stages[0].instructions, Some(10));
    assert_eq!(summary.stages[1].wall_time_ns, 400_000);
    assert_eq!(summary.stages[1].rss_open_gib, Some(3.0));
    assert_eq!(summary.stages[1].rss_delta_gib, Some(2.0));
    assert_eq!(summary.stages[1].instructions, Some(40));
    let root = summary.root.unwrap();
    assert_eq!(root.wall_time_ns, 550_000);
    assert_eq!(root.instructions, Some(60));
}

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

    let (out_path, summary) =
        jolt_profiling::summary::finalize_trace(&trace_path, &fixture_context(), fixture_peak())
            .unwrap();

    assert_eq!(out_path, summary_path(&trace_path));
    assert_eq!(summary.peak_rss_gib, Some(4.0));
    let rewritten: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    assert!(rewritten.iter().all(|e| {
        e.get("args")
            .and_then(Value::as_object)
            .is_none_or(|args| !args.keys().any(|k| k.starts_with("counters.")))
    }));
    assert!(std::fs::read_dir(&dir).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    let reparsed: ProfileSummary =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(reparsed.peak_rss_gib, Some(4.0));
    assert_eq!(reparsed.peak_footprint_gib, Some(6.0));
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
