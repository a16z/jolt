//! Flush-time rendering of a chrome trace into machine-queryable telemetry.
//!
//! One span stream, two renderings: [`finalize_trace`] parses the
//! `tracing-chrome` output after the flush guard drops, rewrites the
//! `counters.*` monitor events into native chrome counter events
//! (`"ph": "C"` — `tracing-chrome` cannot emit them itself, which is what
//! previously forced the offline `postprocess_trace.py` step), writes the
//! trace back, and aggregates the same events into
//! `{trace_name}.summary.json`. Both artifacts therefore derive from one
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
    /// Process-lifetime `getrusage` high-water mark, in GiB. Includes guest
    /// compile / tracer execution, unlike `root.peak_memory_gib`.
    pub peak_rss_gib: Option<f64>,
    /// Per-label aggregates over every span instance on every thread.
    pub spans: BTreeMap<String, SpanAggregate>,
    /// Per-stage rollup in pipeline order (only stages present in the trace).
    pub stages: Vec<StageSummary>,
    /// Per-counter sample statistics (`memory_gib`, `cpu_percent`, …).
    pub counters: BTreeMap<String, CounterSummary>,
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
struct Interval {
    start_us: f64,
    end_us: f64,
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

/// Everything [`build_summary`] needs from one replay of the event stream.
struct TraceAggregate {
    spans: BTreeMap<String, SpanAggregate>,
    root: Option<(Interval, u64)>,
    stage_intervals: Vec<(String, Interval)>,
    counter_points: BTreeMap<String, Vec<(f64, f64)>>,
}

/// Replays the chrome events through per-thread span stacks, producing
/// per-label aggregates, the root/stage intervals, and counter samples.
///
/// Accepts both raw (`counters.*` instant events) and converted (`"ph": "C"`)
/// counter encodings, so it can run before or after
/// [`convert_counter_events`].
fn aggregate_events(events: &[Value], root_span: &str) -> TraceAggregate {
    let mut stacks: HashMap<u64, Vec<OpenSpan>> = HashMap::new();
    let mut spans: BTreeMap<String, SpanAggregate> = BTreeMap::new();
    let mut root: Option<(Interval, u64)> = None;
    let mut stage_intervals: Vec<(String, Interval)> = Vec::new();
    let mut counter_points: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();

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
) -> ProfileSummary {
    let aggregate = aggregate_events(events, taxonomy::ROOT_SPAN);
    let memory_points = aggregate.counter_points.get(MEMORY_COUNTER);

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

    let stages = ordered
        .into_iter()
        .map(|(label, interval)| {
            let row = stage_rows.iter().find(|row| row.stage == label);
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

/// The summary artifact path for a given trace path:
/// `{trace_name}.json` → `{trace_name}.summary.json`.
pub fn summary_path(trace_path: &Path) -> PathBuf {
    trace_path.with_extension("summary.json")
}

/// Flush-time pipeline entry: rewrite counter events in the trace file
/// in place, then aggregate the same events (folding in the drained
/// [`StageMemoryRow`]s and the `getrusage` peak) into
/// `{trace_name}.summary.json` next to it.
///
/// Call after dropping [`TracingGuards`](crate::TracingGuards) — the chrome
/// layer finalizes the trace file on guard drop.
pub fn finalize_trace(
    trace_path: &Path,
    ctx: &SummaryContext,
) -> Result<(PathBuf, ProfileSummary), SummaryError> {
    let events = read_events(trace_path)?;
    let events = convert_counter_events(events);
    let trace_json = serde_json::to_string(&events)?;
    std::fs::write(trace_path, trace_json).map_err(|source| SummaryError::Write {
        path: trace_path.to_path_buf(),
        source,
    })?;

    let timestamp_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let summary = build_summary(
        &events,
        ctx,
        &crate::stage_memory::take_stage_memory_rows(),
        crate::memory::peak_rss_bytes(),
        timestamp_unix_secs,
        git_rev(),
    );

    let out_path = summary_path(trace_path);
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&out_path, summary_json).map_err(|source| SummaryError::Write {
        path: out_path.clone(),
        source,
    })?;
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
