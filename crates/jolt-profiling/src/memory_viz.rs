//! The memory-timeline companion page: one self-contained `memory.html`
//! per allocative profile run, next to the run's trace.
//!
//! One time axis carries the whole memory story — the continuous
//! `memory_gib` counter as the RSS envelope, the stage spans as labeled
//! bands, and at each `heap_snapshot` instant a stacked composition column
//! of the live batch kernels (exact bytes from the `.folded` twins), topped
//! with the gray "unattributed" residual up to the RSS envelope (allocator
//! retention + unvisited allocations). Clicking a column opens the
//! snapshot's full-depth icicle. No external dependencies: the data is
//! inlined as JSON and rendered by inline SVG/JS, so the file opens
//! anywhere.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::summary::{ProfileSummary, SummaryError, TraceAggregate};

/// The page template; `__DATA_JSON__` is replaced by the run's payload.
const TEMPLATE: &str = include_str!("memory_viz.html");

/// Cap on inlined RSS samples — beyond this the series is stride-decimated
/// (the envelope's shape survives; per-sample detail stays in the trace).
const MAX_RSS_POINTS: usize = 1500;

/// Derives `memory.html` next to the trace, mirroring
/// [`summary_path`](crate::summary::summary_path) — the per-run directory
/// keeps the fixed sibling name collision-free.
pub fn memory_viz_path(trace_path: &Path) -> PathBuf {
    trace_path.with_file_name("memory.html")
}

/// Renders and writes the page. Call with the flush-time aggregate of the
/// same event stream the summary was built from — both renderings derive
/// from one stream by construction.
pub(crate) fn write_memory_viz(
    trace_path: &Path,
    summary: &ProfileSummary,
    aggregate: &TraceAggregate,
    folded: &BTreeMap<String, String>,
) -> Result<PathBuf, SummaryError> {
    let Some((root, _)) = aggregate.root else {
        return Ok(memory_viz_path(trace_path));
    };
    let to_rel_s = |ts_us: f64| ((ts_us - root.start_us) / 1e6).max(0.0);

    let stages: Vec<Value> = aggregate
        .stage_intervals
        .iter()
        .map(|(label, interval)| {
            json!({
                "label": label,
                "start_s": to_rel_s(interval.start_us),
                "end_s": to_rel_s(interval.end_us),
            })
        })
        .collect();

    let rss_full: Vec<(f64, f64)> = aggregate
        .counter_points
        .get("memory_gib")
        .map(|points| {
            points
                .iter()
                .map(|(ts, gib)| (to_rel_s(*ts), *gib))
                .collect()
        })
        .unwrap_or_default();
    let stride = rss_full.len().div_ceil(MAX_RSS_POINTS).max(1);
    let rss: Vec<Value> = rss_full
        .iter()
        .step_by(stride)
        .map(|(t, gib)| json!([t, gib]))
        .collect();

    let snapshots: Vec<Value> = summary
        .heap
        .iter()
        .map(|(label, snapshot)| {
            json!({
                "label": label,
                "at_s": snapshot.at_ns.map(|ns| ns as f64 / 1e9),
                "total_bytes": snapshot.total_bytes,
                "roots": snapshot.roots,
                "folded": folded.get(label).map_or("", String::as_str),
            })
        })
        .collect();

    let payload = json!({
        // Page title: the run directory's name ({timestamp}_{trace_name})
        // identifies both the workload and the specific run.
        "trace_name": trace_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or("trace"),
        "run": summary.run,
        "peak_rss_gib": summary.peak_rss_gib,
        "duration_s": to_rel_s(root.end_us),
        "stages": stages,
        "rss": rss,
        "snapshots": snapshots,
    });

    let html = TEMPLATE.replace("__DATA_JSON__", &serde_json::to_string(&payload)?);
    let out_path = memory_viz_path(trace_path);
    crate::summary::write_atomic(&out_path, &html)?;
    Ok(out_path)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::summary::{aggregate_events, build_summary, parse_folded, SummaryContext};

    #[test]
    fn writes_a_self_contained_page_with_the_run_inlined() {
        let events: Vec<Value> = serde_json::from_value(json!([
            { "ph": "B", "pid": 1, "name": "jolt_prover::prove", "tid": 1, "ts": 0.0 },
            { "ph": "B", "pid": 1, "name": "prove_stage1", "tid": 1, "ts": 100.0 },
            { "ph": "i", "pid": 1, "name": "heap_snapshot", "tid": 1, "ts": 250.0,
              "args": { "snapshot": "Stage1Batch_prepared" } },
            { "ph": "i", "pid": 1, "name": "monitor.rs:70", "tid": 3, "ts": 300.0,
              "args": { "counters.memory_gib": 2.0 } },
            { "ph": "E", "pid": 1, "name": "prove_stage1", "tid": 1, "ts": 900.0 },
            { "ph": "E", "pid": 1, "name": "jolt_prover::prove", "tid": 1, "ts": 1000.0 },
        ]))
        .unwrap();
        let folded: BTreeMap<String, String> = [(
            "Stage1Batch_prepared".to_string(),
            "KernelA;opening_tables 4096\nProofSession 48\n".to_string(),
        )]
        .into_iter()
        .collect();
        let heap = folded
            .iter()
            .map(|(label, text)| (label.clone(), parse_folded(text)))
            .collect();
        let ctx = SummaryContext {
            workload: "fibonacci".to_string(),
            scale_log2: 13,
            backend: "reference".to_string(),
        };
        let summary = build_summary(&events, &ctx, &[], Some(1 << 30), 0, None, heap);
        let aggregate = aggregate_events(&events, crate::taxonomy::ROOT_SPAN);

        let run_dir =
            std::env::temp_dir().join(format!("jolt_memory_viz_test_{}", std::process::id()));
        std::fs::create_dir_all(&run_dir).unwrap();
        let trace_path = run_dir.join("trace.json");
        let out = write_memory_viz(&trace_path, &summary, &aggregate, &folded).unwrap();
        let html = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_dir_all(&run_dir);

        assert_eq!(
            summary.heap["Stage1Batch_prepared"].at_ns,
            Some(250_000),
            "snapshot instant joined by label"
        );
        assert_eq!(out, run_dir.join("memory.html"));
        assert!(!html.contains("__DATA_JSON__"), "payload was substituted");
        assert!(html.contains("Stage1Batch_prepared"), "snapshot inlined");
        assert!(html.contains("\"at_s\":"), "snapshot instant inlined");
        assert!(
            html.contains("KernelA;opening_tables"),
            "folded tree inlined"
        );
        assert!(html.contains("renderTimeline"), "renderer inlined");
    }
}
