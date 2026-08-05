//! Flush-time rendering of a chrome trace into machine-queryable telemetry.
//!
//! One span stream, two renderings: [`finalize_trace`] parses the
//! `tracing-chrome` output after the flush guard drops, rewrites the
//! `counters.*` monitor events into native chrome counter events
//! (`"ph": "C"` — `tracing-chrome` cannot emit them itself, which is what
//! previously forced the offline `postprocess_trace.py` step), writes the
//! trace back, and aggregates the same events into
//! `summary.json` next to the trace. Both artifacts therefore derive from one
//! event stream by construction.
//!
//! The serde structs here are the normative summary schema, mirrored by the
//! checked-in JSON Schema (`schema/summary.schema.json`, drift-locked by a
//! fixture test) and versioned via [`SUMMARY_SCHEMA_VERSION`].
//!
//! # Aggregation semantics under rayon parallelism
//!
//! - *Self time* of a span = inclusive duration minus the union of its
//!   same-thread children's intervals. Same-thread children nest strictly
//!   (stack discipline), so the union is a plain sum. Spans opened on other
//!   rayon worker threads attribute to their own labels but never subtract
//!   from a parent on a different thread.
//! - *Dark time* = the root span's self time on the root thread's timeline:
//!   wallclock not covered by any depth-1 child.
//! - *Per-label totals* sum inclusive durations across all instances on all
//!   threads and may legitimately exceed wallclock under parallelism.
//! - *Per-stage / overall peak memory* = max over `memory_gib` counter
//!   samples falling inside the stage / root span's interval; `null` when
//!   the interval contains no samples (short stages at the monitor's
//!   minimum 50 ms sampling interval).
//! - *Per-stage boundary RSS* comes from [`StageMemoryRow`] (retained growth
//!   per stage, deliberately not within-stage peak); *headline peak RSS* is
//!   the `getrusage` high-water mark, which sampling cannot miss.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::stage_memory::StageMemoryRow;
use crate::taxonomy;
use crate::units::BYTES_PER_GIB;

/// Version of the `summary.json` schema emitted by this module.
pub const SUMMARY_SCHEMA_VERSION: u32 = 1;

/// The checked-in JSON Schema mirroring [`ProfileSummary`]. A unit test
/// asserts it matches the schemars-generated schema, so edits to the structs
/// force a matching schema-file update.
pub const SUMMARY_SCHEMA_JSON: &str = include_str!("../schema/summary.schema.json");

#[derive(Debug, thiserror::Error)]
pub enum SummaryError {
    #[error("reading trace {path}: {source}")]
    ReadTrace {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parsing trace {path}: {source}")]
    ParseTrace {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("trace {path} is not a chrome event array")]
    NotAnEventArray { path: PathBuf },
    #[error("writing {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("serializing json: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// The machine-queryable flush-time summary — the stable-schema artifact
/// `jolt-eval` telemetry objectives and quick `jq` queries consume.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileSummary {
    pub schema_version: u32,
    pub taxonomy_version: u32,
    pub run: RunMetadata,
    /// `null` if the trace contains no root span (see
    /// [`taxonomy::ROOT_SPAN`]).
    pub root: Option<RootSummary>,
    /// Process-lifetime `getrusage` high-water mark, in GiB, sampled by the
    /// harness right after the workload — before the flush-time trace
    /// parse/rewrite can inflate it. Includes guest compile / tracer
    /// execution, unlike `root.peak_memory_gib`.
    pub peak_rss_gib: Option<f64>,
    /// Per-label aggregates over every span instance on every thread.
    pub spans: BTreeMap<String, SpanAggregate>,
    /// Per-stage rollup in pipeline order (only stages present in the trace).
    pub stages: Vec<StageSummary>,
    /// Per-counter sample statistics (`memory_gib`, `cpu_percent`, …).
    pub counters: BTreeMap<String, CounterSummary>,
    /// Heap attribution from the allocative lane's mid-stage snapshots,
    /// keyed by snapshot label (e.g. `Stage2Batch_prepared`). Empty unless
    /// the run was profiled with the `allocative` feature. Exact bytes,
    /// parsed from the `.folded` snapshots in the run directory; full stack
    /// detail stays in those files and renders in the run's `memory.html`.
    #[serde(default)]
    pub heap: BTreeMap<String, HeapSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunMetadata {
    pub workload: String,
    /// log2 of the padded trace length the workload was scaled to.
    pub scale_log2: u32,
    pub backend: String,
    pub timestamp_unix_secs: u64,
    pub git_rev: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RootSummary {
    pub label: String,
    pub wall_time_ns: u64,
    /// Root wallclock not covered by any depth-1 child span on the root
    /// thread.
    pub dark_time_ns: u64,
    pub dark_time_fraction: f64,
    /// Max `memory_gib` sample inside the root span's interval.
    pub peak_memory_gib: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpanAggregate {
    /// Number of span instances with this label.
    pub count: u64,
    /// Inclusive time summed over all instances (may exceed wallclock under
    /// parallelism).
    pub total_ns: u64,
    /// Exclusive time: inclusive minus same-thread children, summed over all
    /// instances.
    pub self_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StageSummary {
    pub label: String,
    pub wall_time_ns: u64,
    /// Boundary RSS from [`StageMemoryRow`]: process RSS when the stage span
    /// opened / closed (retained growth, not within-stage peak).
    pub rss_open_gib: Option<f64>,
    pub rss_close_gib: Option<f64>,
    pub rss_delta_gib: Option<f64>,
    /// Max `memory_gib` sample inside the stage span's interval; `null` when
    /// the stage closed between monitor samples.
    pub peak_memory_gib: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CounterSummary {
    pub samples: u64,
    pub max: f64,
    pub mean: f64,
}

/// One allocative mid-stage snapshot's heap attribution.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeapSnapshot {
    /// Total live bytes across every visited root at the snapshot.
    pub total_bytes: u64,
    /// Live bytes per root frame — the top level of the folded stacks: the
    /// member kernels' concrete type names (which name the relation) and
    /// `ProofSession`.
    pub roots: BTreeMap<String, u64>,
    /// When the snapshot fired, in ns since the root span opened — joined
    /// from the driver's `heap_snapshot` instant event. `null` when the
    /// event or the root span is missing (e.g. traces from before the event
    /// existed).
    #[serde(default)]
    pub at_ns: Option<u64>,
}

/// Parses one folded-stacks blob (`root;child;… BYTES` per line, the
/// flamegraph interchange format the allocative lane persists) into
/// per-root totals. Malformed lines are skipped.
pub fn parse_folded(folded: &str) -> HeapSnapshot {
    let mut roots: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_bytes = 0u64;
    for line in folded.lines() {
        let Some((stack, bytes)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(bytes) = bytes.parse::<u64>() else {
            continue;
        };
        let root = stack.split(';').next().unwrap_or(stack);
        *roots.entry(root.to_string()).or_default() += bytes;
        total_bytes += bytes;
    }
    HeapSnapshot {
        total_bytes,
        roots,
        at_ns: None,
    }
}

/// Reads every `{prefix}<label>.folded` blob the allocative lane left,
/// keyed by `<label>`. `prefix` is the same path-string prefix the
/// flamegraph writer uses — a bare run directory (`{run_dir}/`) in the
/// per-run layout, where the empty file-name part matches every `.folded`
/// file in the directory. The raw text feeds both the summary's heap
/// section (via [`parse_folded`]) and the memory-timeline viz's full-depth
/// icicles.
#[cfg(feature = "allocative")]
fn read_folded_files(prefix: &str) -> BTreeMap<String, String> {
    let mut snapshots = BTreeMap::new();
    // Split on the last separator by string, not Path methods: a trailing
    // `/` means "directory + empty stem", which `Path::file_name` would
    // misread as the directory's own name.
    let (dir, stem) = prefix.rsplit_once('/').unwrap_or((".", prefix));
    let dir = Path::new(if dir.is_empty() { "." } else { dir });
    let Ok(entries) = std::fs::read_dir(dir) else {
        return snapshots;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(label) = name
            .to_str()
            .and_then(|name| name.strip_prefix(stem))
            .and_then(|name| name.strip_suffix(".folded"))
        else {
            continue;
        };
        if let Ok(folded) = std::fs::read_to_string(entry.path()) {
            let _ = snapshots.insert(label.to_string(), folded);
        }
    }
    snapshots
}

/// Run identity threaded into [`RunMetadata`] by the profile harness.
#[derive(Clone, Debug)]
pub struct SummaryContext {
    pub workload: String,
    pub scale_log2: u32,
    pub backend: String,
}

/// The `memory_gib` counter name both peak-memory metrics sample.
const MEMORY_COUNTER: &str = "memory_gib";

/// Rewrites every event carrying `counters.*` args into one chrome counter
/// event (`"ph": "C"`) per counter, dropping the original event. All other
/// events pass through unchanged.
pub fn convert_counter_events(events: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(events.len());
    for event in events {
        let Some(samples) = counter_samples(&event) else {
            out.push(event);
            continue;
        };
        let (ts, pid, tid) = (
            event.get("ts").cloned().unwrap_or(Value::Null),
            event.get("pid").cloned().unwrap_or(Value::Null),
            event.get("tid").cloned().unwrap_or(Value::Null),
        );
        for (name, value) in samples {
            out.push(serde_json::json!({
                "name": name,
                "ph": "C",
                "ts": ts,
                "pid": pid,
                "tid": tid,
                "args": { name: value },
            }));
        }
    }
    out
}

/// Numeric arg value, tolerating `tracing-chrome`'s stringified field
/// encoding (`"counters.memory_gib": "0.0074"`).
fn json_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Extracts `counters.*` args from an event, prefix stripped. `None` when
/// the event carries no counter fields (i.e. it is not a monitor sample).
fn counter_samples(event: &Value) -> Option<Vec<(String, f64)>> {
    let obj = event.as_object()?;
    // Span begin/end and metadata events never carry counter fields; only
    // instant events (the monitor's `tracing::debug!` samples) do.
    match obj.get("ph").and_then(Value::as_str) {
        Some("i" | "I") => {}
        _ => return None,
    }
    let args = obj.get("args")?.as_object()?;
    let samples: Vec<(String, f64)> = args
        .iter()
        .filter_map(|(k, v)| {
            let name = k.strip_prefix("counters.")?;
            Some((name.to_string(), json_f64(v)?))
        })
        .collect();
    (!samples.is_empty()).then_some(samples)
}

/// One thread's currently open span during the stack replay.
struct OpenSpan {
    name: String,
    start_us: f64,
    child_us: f64,
}

/// Half-open microsecond interval of one closed span instance.
#[derive(Clone, Copy)]
pub(crate) struct Interval {
    pub(crate) start_us: f64,
    pub(crate) end_us: f64,
}

impl Interval {
    fn contains(&self, ts_us: f64) -> bool {
        ts_us >= self.start_us && ts_us <= self.end_us
    }

    fn duration_ns(&self) -> u64 {
        us_to_ns(self.end_us - self.start_us)
    }
}

fn us_to_ns(us: f64) -> u64 {
    if us <= 0.0 {
        0
    } else {
        (us * 1_000.0).round() as u64
    }
}

/// Everything [`build_summary`] (and the memory-timeline viz) needs from
/// one replay of the event stream.
pub(crate) struct TraceAggregate {
    pub(crate) spans: BTreeMap<String, SpanAggregate>,
    pub(crate) root: Option<(Interval, u64)>,
    pub(crate) stage_intervals: Vec<(String, Interval)>,
    pub(crate) counter_points: BTreeMap<String, Vec<(f64, f64)>>,
    /// `heap_snapshot` instant events: snapshot label → absolute trace µs.
    pub(crate) snapshot_instants: Vec<(String, f64)>,
}

/// Replays the chrome events through per-thread span stacks, producing
/// per-label aggregates, the root/stage intervals, and counter samples.
///
/// Accepts both raw (`counters.*` instant events) and converted (`"ph": "C"`)
/// counter encodings, so it can run before or after
/// [`convert_counter_events`].
pub(crate) fn aggregate_events(events: &[Value], root_span: &str) -> TraceAggregate {
    let mut stacks: HashMap<u64, Vec<OpenSpan>> = HashMap::new();
    let mut spans: BTreeMap<String, SpanAggregate> = BTreeMap::new();
    let mut root: Option<(Interval, u64)> = None;
    let mut stage_intervals: Vec<(String, Interval)> = Vec::new();
    let mut counter_points: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut snapshot_instants: Vec<(String, f64)> = Vec::new();

    for event in events {
        let Some(obj) = event.as_object() else {
            continue;
        };
        let ph = obj.get("ph").and_then(Value::as_str).unwrap_or_default();
        let name = obj.get("name").and_then(Value::as_str).unwrap_or_default();
        let ts = obj.get("ts").and_then(Value::as_f64).unwrap_or_default();
        let tid = obj.get("tid").and_then(Value::as_u64).unwrap_or_default();

        match ph {
            "B" => stacks.entry(tid).or_default().push(OpenSpan {
                name: name.to_string(),
                start_us: ts,
                child_us: 0.0,
            }),
            "E" => {
                let Some(stack) = stacks.get_mut(&tid) else {
                    continue;
                };
                // Chrome B/E events nest strictly per thread; a name mismatch
                // means a corrupt or truncated trace, in which case the
                // unmatched opens are dropped rather than misattributed.
                let Some(matching) = stack.iter().rposition(|open| open.name == name) else {
                    continue;
                };
                stack.truncate(matching + 1);
                let open = match stack.pop() {
                    Some(open) => open,
                    None => continue,
                };
                let interval = Interval {
                    start_us: open.start_us,
                    end_us: ts,
                };
                let dur_us = (ts - open.start_us).max(0.0);
                let self_us = (dur_us - open.child_us).max(0.0);
                if let Some(parent) = stack.last_mut() {
                    parent.child_us += dur_us;
                }

                let entry = spans.entry(open.name.clone()).or_insert(SpanAggregate {
                    count: 0,
                    total_ns: 0,
                    self_ns: 0,
                });
                entry.count += 1;
                entry.total_ns += us_to_ns(dur_us);
                entry.self_ns += us_to_ns(self_us);

                if open.name == root_span {
                    // Keep the longest instance if the label somehow repeats.
                    let dark_ns = us_to_ns(self_us);
                    if root.is_none_or(|(prev, _)| interval.duration_ns() > prev.duration_ns()) {
                        root = Some((interval, dark_ns));
                    }
                }
                if open.name.starts_with("prove_stage") {
                    stage_intervals.push((open.name, interval));
                }
            }
            "C" => {
                if let Some(args) = obj.get("args").and_then(Value::as_object) {
                    for (key, value) in args {
                        if let Some(value) = json_f64(value) {
                            counter_points
                                .entry(key.clone())
                                .or_default()
                                .push((ts, value));
                        }
                    }
                }
            }
            _ => {
                // The driver's `heap_snapshot` instant events carry the
                // snapshot label in their `snapshot` field. tracing-chrome
                // records the `&str` through `Debug`, so the value arrives
                // quote-wrapped — trim before joining.
                if let Some(label) = obj
                    .get("args")
                    .and_then(Value::as_object)
                    .and_then(|args| args.get("snapshot"))
                    .and_then(Value::as_str)
                {
                    snapshot_instants.push((label.trim_matches('"').to_string(), ts));
                }
                if let Some(samples) = counter_samples(event) {
                    for (key, value) in samples {
                        counter_points.entry(key).or_default().push((ts, value));
                    }
                }
            }
        }
    }

    TraceAggregate {
        spans,
        root,
        stage_intervals,
        counter_points,
        snapshot_instants,
    }
}

fn peak_within(points: Option<&Vec<(f64, f64)>>, interval: Interval) -> Option<f64> {
    points?
        .iter()
        .filter(|(ts, _)| interval.contains(*ts))
        .map(|(_, value)| *value)
        .reduce(f64::max)
}

/// Pure assembly of the summary from parsed chrome events plus the
/// out-of-band memory bookkeeping. The I/O wrapper is [`finalize_trace`];
/// fixture tests call this directly.
pub fn build_summary(
    events: &[Value],
    ctx: &SummaryContext,
    stage_rows: &[StageMemoryRow],
    peak_rss_bytes: Option<u64>,
    timestamp_unix_secs: u64,
    git_rev: Option<String>,
    mut heap: BTreeMap<String, HeapSnapshot>,
) -> ProfileSummary {
    let aggregate = aggregate_events(events, taxonomy::ROOT_SPAN);
    let memory_points = aggregate.counter_points.get(MEMORY_COUNTER);

    // Situate the allocative snapshots on the trace clock: join the
    // driver's `heap_snapshot` instant events by label, as ns since the
    // root span opened.
    if let Some((root_interval, _)) = aggregate.root {
        for (label, ts_us) in &aggregate.snapshot_instants {
            if let Some(snapshot) = heap.get_mut(label) {
                snapshot.at_ns = Some(us_to_ns((ts_us - root_interval.start_us).max(0.0)));
            }
        }
    }

    let root = aggregate.root.map(|(interval, dark_ns)| {
        let wall_ns = interval.duration_ns();
        RootSummary {
            label: taxonomy::ROOT_SPAN.to_string(),
            wall_time_ns: wall_ns,
            dark_time_ns: dark_ns,
            dark_time_fraction: if wall_ns == 0 {
                0.0
            } else {
                dark_ns as f64 / wall_ns as f64
            },
            peak_memory_gib: peak_within(memory_points, interval),
        }
    });

    // Pipeline order per the taxonomy, then any unknown prove_stage* labels
    // (e.g. legacy stage spans in a mixed trace) in first-seen order.
    let mut ordered: Vec<(String, Interval)> = Vec::with_capacity(aggregate.stage_intervals.len());
    for stage in taxonomy::STAGE_SPANS {
        ordered.extend(
            aggregate
                .stage_intervals
                .iter()
                .filter(|(label, _)| label == stage)
                .cloned(),
        );
    }
    ordered.extend(
        aggregate
            .stage_intervals
            .iter()
            .filter(|(label, _)| !taxonomy::STAGE_SPANS.contains(&label.as_str()))
            .cloned(),
    );

    // Stage intervals and StageMemoryRows are both recorded in span-close
    // order, so the i-th interval of a label pairs with the i-th row of
    // that label — occurrence-indexed, not first-match, so a trace with
    // repeated stage labels (e.g. a future multi-prove harness) can't
    // attach the first prove's boundary RSS to every instance. `ordered`
    // preserves within-label close order by construction (stable filters).
    let mut label_occurrence: HashMap<String, usize> = HashMap::new();
    let stages = ordered
        .into_iter()
        .map(|(label, interval)| {
            let occurrence = label_occurrence.entry(label.clone()).or_insert(0);
            let row = stage_rows
                .iter()
                .filter(|row| row.stage == label)
                .nth(*occurrence);
            *occurrence += 1;
            let open = row.map(|r| r.rss_open_bytes as f64 / BYTES_PER_GIB);
            let close = row.map(|r| r.rss_close_bytes as f64 / BYTES_PER_GIB);
            StageSummary {
                wall_time_ns: interval.duration_ns(),
                rss_open_gib: open,
                rss_close_gib: close,
                rss_delta_gib: close.zip(open).map(|(c, o)| c - o),
                peak_memory_gib: peak_within(memory_points, interval),
                label,
            }
        })
        .collect();

    let counters = aggregate
        .counter_points
        .iter()
        .map(|(name, points)| {
            let samples = points.len() as u64;
            let max = points.iter().map(|(_, v)| *v).fold(f64::MIN, f64::max);
            let sum: f64 = points.iter().map(|(_, v)| *v).sum();
            (
                name.clone(),
                CounterSummary {
                    samples,
                    max,
                    mean: sum / samples as f64,
                },
            )
        })
        .collect();

    ProfileSummary {
        schema_version: SUMMARY_SCHEMA_VERSION,
        taxonomy_version: taxonomy::TAXONOMY_VERSION,
        run: RunMetadata {
            workload: ctx.workload.clone(),
            scale_log2: ctx.scale_log2,
            backend: ctx.backend.clone(),
            timestamp_unix_secs,
            git_rev,
        },
        root,
        peak_rss_gib: peak_rss_bytes.map(|bytes| bytes as f64 / BYTES_PER_GIB),
        spans: aggregate.spans,
        stages,
        counters,
        heap,
    }
}

/// Parses a chrome trace file (bare event array or `{"traceEvents": [...]}`).
fn read_events(path: &Path) -> Result<Vec<Value>, SummaryError> {
    let data = std::fs::read_to_string(path).map_err(|source| SummaryError::ReadTrace {
        path: path.to_path_buf(),
        source,
    })?;
    let parsed: Value = serde_json::from_str(&data).map_err(|source| SummaryError::ParseTrace {
        path: path.to_path_buf(),
        source,
    })?;
    match parsed {
        Value::Array(events) => Ok(events),
        Value::Object(mut obj) => match obj.remove("traceEvents") {
            Some(Value::Array(events)) => Ok(events),
            _ => Err(SummaryError::NotAnEventArray {
                path: path.to_path_buf(),
            }),
        },
        _ => Err(SummaryError::NotAnEventArray {
            path: path.to_path_buf(),
        }),
    }
}

/// The summary artifact path for a given trace path: `summary.json` next to
/// the trace. The trace lives in a per-run directory, so a fixed sibling
/// name cannot collide across runs.
pub fn summary_path(trace_path: &Path) -> PathBuf {
    trace_path.with_file_name("summary.json")
}

/// Atomic file replacement: write `{path}.tmp`, then rename over `path`.
/// `fs::write` alone truncates first, so a crash mid-write would destroy the
/// existing artifact — for a trace that took hours to produce, cheap
/// insurance.
pub(crate) fn write_atomic(path: &Path, data: &str) -> Result<(), SummaryError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let io = std::fs::write(&tmp, data).and_then(|()| std::fs::rename(&tmp, path));
    io.map_err(|source| SummaryError::Write {
        path: path.to_path_buf(),
        source,
    })
}

/// Flush-time pipeline entry: rewrite counter events in the trace file
/// (atomically — temp file + rename, never a truncating in-place write),
/// then aggregate the same events (folding in the drained
/// [`StageMemoryRow`]s and the caller-captured `getrusage` peak) into
/// `summary.json` next to it.
///
/// Call after dropping [`TracingGuards`](crate::TracingGuards) — the chrome
/// layer finalizes the trace file on guard drop. `peak_rss_bytes` must be
/// sampled by the caller right after the workload (see
/// [`peak_rss_bytes`](crate::memory::peak_rss_bytes)): sampling here would
/// report this function's own trace parse/expand allocations — tooling
/// memory, not the profiled workload — whenever they exceed the prove's
/// footprint.
pub fn finalize_trace(
    trace_path: &Path,
    ctx: &SummaryContext,
    peak_rss_bytes: Option<u64>,
) -> Result<(PathBuf, ProfileSummary), SummaryError> {
    let events = read_events(trace_path)?;
    let events = convert_counter_events(events);
    let trace_json = serde_json::to_string(&events)?;
    write_atomic(trace_path, &trace_json)?;

    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    // Heap attribution from the allocative lane's mid-stage snapshots, if
    // the harness opted in (prefix set + .folded twins on disk). The raw
    // folded text is kept for the memory-timeline viz's icicles.
    #[cfg(feature = "allocative")]
    let folded_files = crate::flamegraph::flamegraph_prefix()
        .map(read_folded_files)
        .unwrap_or_default();
    #[cfg(not(feature = "allocative"))]
    let folded_files: BTreeMap<String, String> = BTreeMap::new();
    let heap = folded_files
        .iter()
        .map(|(label, folded)| (label.clone(), parse_folded(folded)))
        .collect();

    let summary = build_summary(
        &events,
        ctx,
        &crate::stage_memory::take_stage_memory_rows(),
        peak_rss_bytes,
        timestamp_unix_secs,
        git_rev(),
        heap,
    );

    let out_path = summary_path(trace_path);
    let summary_json = serde_json::to_string_pretty(&summary)?;
    write_atomic(&out_path, &summary_json)?;

    // The memory-timeline companion (`memory.html`): only
    // meaningful when the allocative lane produced snapshots and the trace
    // has a root span to anchor time.
    if !summary.heap.is_empty() && summary.root.is_some() {
        let aggregate = aggregate_events(&events, taxonomy::ROOT_SPAN);
        let _ =
            crate::memory_viz::write_memory_viz(trace_path, &summary, &aggregate, &folded_files)?;
    }
    Ok((out_path, summary))
}

/// Short git revision of the working directory, if resolvable.
fn git_rev() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
