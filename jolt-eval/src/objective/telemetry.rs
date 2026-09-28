//! String-keyed telemetry objectives over the modular prover's
//! `summary.json` (`specs/prover-telemetry.md`).
//!
//! Key grammar, pinned:
//!
//! ```text
//! telemetry:<workload>:<metric>
//! <workload> ::= [a-z0-9-]+            (case-sensitive; must be in WORKLOAD_SCALES)
//! <metric>   ::= prover_time_s         (root-span duration, seconds)
//!              | peak_rss_gib          (process-lifetime getrusage high-water mark)
//!              | peak_memory_gib       (max over memory samples in the root span)
//!              | total:<span-label>    (inclusive time summed over all instances, seconds)
//!              | self:<span-label>     (exclusive time summed over all instances, seconds)
//!              | heap:<snapshot>       (allocative snapshot total, exact bytes)
//!              | heap:<snapshot>:<root> (one root frame's bytes; root is verbatim)
//! ```
//!
//! Parsing splits on the first three `:` only — everything after the third
//! colon is the **verbatim span label and may itself contain `:`** (e.g.
//! `telemetry:sha2-chain:total:EqPolynomial::evals`). A key referencing a
//! label absent from `summary.json` is a measurement error, never `0.0` —
//! silent zeros would corrupt optimizer accept/reject decisions.
//!
//! An optimization agent can therefore target any span it discovers in a
//! trace without editing `jolt-eval`; parsed keys are interned
//! (`Box::leak`) so the objective stays `Copy` like every other
//! [`OptimizationObjective`] variant.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

use super::{MeasurementError, OptimizationObjective};

/// The normative measurement-scale table: `measure` always invokes the
/// profile bin with an explicit `--scale` from here (initialized to the
/// spec's workload-table defaults).
pub const WORKLOAD_SCALES: &[(&str, u32)] = &[
    ("fibonacci", 16),
    ("sha2-chain", 22),
    ("sha3-chain", 22),
    ("btreemap", 20),
];

/// The `summary.json` schema version this parser understands.
const SUPPORTED_SCHEMA_VERSION: u64 = 1;

/// Which summary field a telemetry key reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TelemetryMetric {
    /// Root-span duration, seconds.
    ProverTimeS,
    /// Process-lifetime `getrusage` high-water mark, GiB.
    PeakRssGib,
    /// Max over memory samples inside the root span, GiB.
    PeakMemoryGib,
    /// Inclusive time of one span label summed over all instances, seconds.
    Total(&'static str),
    /// Exclusive time of one span label summed over all instances, seconds.
    SelfTime(&'static str),
    /// Live bytes in one allocative mid-stage snapshot (e.g.
    /// `Stage2Batch_prepared`): the whole snapshot, or one root frame (a
    /// kernel type name, verbatim — it may contain `:`) within it.
    Heap {
        snapshot: &'static str,
        root: Option<&'static str>,
    },
}

/// One parsed `telemetry:<workload>:<metric>` objective.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TelemetryObjective {
    /// The full verbatim key (also the objective's CLI name).
    pub key: &'static str,
    pub workload: &'static str,
    pub metric: TelemetryMetric,
}

fn intern(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Const constructor for curated fibonacci-workload keys (the cheapest
/// workload — the sensible default for optimizer loops).
const fn fibonacci(key: &'static str, metric: TelemetryMetric) -> OptimizationObjective {
    OptimizationObjective::Telemetry(TelemetryObjective {
        key,
        workload: "fibonacci",
        metric,
    })
}

/// Curated defaults (spec-pinned) so `optimize --list` stays useful; any
/// other span is reachable through the parameterized key grammar.
pub const MODULAR_PROVER_TIME: OptimizationObjective = fibonacci(
    "telemetry:fibonacci:prover_time_s",
    TelemetryMetric::ProverTimeS,
);
pub const MODULAR_COMMIT_TIME: OptimizationObjective = fibonacci(
    "telemetry:fibonacci:total:commit_witness",
    TelemetryMetric::Total("commit_witness"),
);
pub const MODULAR_ROUND_LOOP_TIME: OptimizationObjective = fibonacci(
    "telemetry:fibonacci:total:prove_batch",
    TelemetryMetric::Total("prove_batch"),
);
/// Per-stage inclusive totals, pipeline order.
pub const MODULAR_STAGE_TOTALS: [OptimizationObjective; 10] = [
    fibonacci(
        "telemetry:fibonacci:total:prove_stage0",
        TelemetryMetric::Total("prove_stage0"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage1",
        TelemetryMetric::Total("prove_stage1"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage2",
        TelemetryMetric::Total("prove_stage2"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage3",
        TelemetryMetric::Total("prove_stage3"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage4",
        TelemetryMetric::Total("prove_stage4"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage5",
        TelemetryMetric::Total("prove_stage5"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage6a",
        TelemetryMetric::Total("prove_stage6a"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage6b",
        TelemetryMetric::Total("prove_stage6b"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage7",
        TelemetryMetric::Total("prove_stage7"),
    ),
    fibonacci(
        "telemetry:fibonacci:total:prove_stage8",
        TelemetryMetric::Total("prove_stage8"),
    ),
];

impl TelemetryObjective {
    /// Parses a `telemetry:<workload>:<metric>` key. Interns the parts, so
    /// repeated parses of long-lived agent loops should reuse the result.
    pub fn parse(key: &str) -> Result<Self, MeasurementError> {
        let rest = key
            .strip_prefix("telemetry:")
            .ok_or_else(|| MeasurementError::new(format!("not a telemetry key: {key}")))?;
        let (workload, metric_str) = rest.split_once(':').ok_or_else(|| {
            MeasurementError::new(format!(
                "malformed telemetry key {key}: expected telemetry:<workload>:<metric>"
            ))
        })?;
        if !WORKLOAD_SCALES.iter().any(|(name, _)| *name == workload) {
            return Err(MeasurementError::new(format!(
                "unknown workload {workload:?} in {key}: expected one of {:?}",
                WORKLOAD_SCALES.iter().map(|(n, _)| *n).collect::<Vec<_>>()
            )));
        }
        let metric = match metric_str {
            "prover_time_s" => TelemetryMetric::ProverTimeS,
            "peak_rss_gib" => TelemetryMetric::PeakRssGib,
            "peak_memory_gib" => TelemetryMetric::PeakMemoryGib,
            other => {
                if let Some(label) = other.strip_prefix("total:") {
                    TelemetryMetric::Total(intern(label))
                } else if let Some(label) = other.strip_prefix("self:") {
                    TelemetryMetric::SelfTime(intern(label))
                } else if let Some(rest) = other.strip_prefix("heap:") {
                    let (snapshot, root) = match rest.split_once(':') {
                        Some((snapshot, root)) => (snapshot, Some(root)),
                        None => (rest, None),
                    };
                    if snapshot.is_empty() {
                        return Err(MeasurementError::new(format!(
                            "empty heap snapshot label in {key}"
                        )));
                    }
                    TelemetryMetric::Heap {
                        snapshot: intern(snapshot),
                        root: root.map(intern),
                    }
                } else {
                    return Err(MeasurementError::new(format!(
                        "unknown telemetry metric {other:?} in {key}"
                    )));
                }
            }
        };
        Ok(Self {
            key: intern(key),
            workload: intern(workload),
            metric,
        })
    }

    pub fn name(&self) -> &str {
        self.key
    }

    pub fn description(&self) -> String {
        format!(
            "modular prover telemetry ({}, scale 2^{})",
            self.key,
            self.scale()
        )
    }

    pub fn units(&self) -> Option<&str> {
        match self.metric {
            TelemetryMetric::ProverTimeS
            | TelemetryMetric::Total(_)
            | TelemetryMetric::SelfTime(_) => Some("s"),
            TelemetryMetric::PeakRssGib | TelemetryMetric::PeakMemoryGib => Some("GiB"),
            TelemetryMetric::Heap { .. } => Some("bytes"),
        }
    }

    /// Heap metrics read the allocative lane's snapshots, so their profile
    /// run must build with the `allocative` feature.
    pub fn needs_allocative(&self) -> bool {
        matches!(self.metric, TelemetryMetric::Heap { .. })
    }

    /// The explicit scale `measure` passes to the profile bin.
    pub fn scale(&self) -> u32 {
        WORKLOAD_SCALES
            .iter()
            .find(|(name, _)| *name == self.workload)
            .map_or(16, |(_, scale)| *scale)
    }

    /// The trace name this objective's profile run uses (also the suffix of
    /// its per-run directories and `latest_` link).
    fn trace_name(&self) -> String {
        format!(
            "modular_{}_{}",
            self.workload.replace('-', "_"),
            self.scale()
        )
    }

    /// The summary artifact path a profile run leaves under `work_dir` —
    /// deterministic because this objective chose the workload and scale,
    /// and because the harness flips the `latest_{trace_name}` link to the
    /// run's timestamped directory only on success.
    pub fn summary_path(&self, work_dir: &Path) -> std::path::PathBuf {
        work_dir.join(format!(
            "benchmark-runs/latest_{}/summary.json",
            self.trace_name()
        ))
    }

    /// Runs the profile bin as a subprocess in `work_dir` (never in-process:
    /// the measurement must compile against the *modified* worktree, and the
    /// artifact path is cwd-relative by design). One run serves every
    /// objective sharing this workload — see [`Self::extract_from_dir`].
    ///
    /// Any `latest_` link a previous run left is removed first, so a failed
    /// run can never expose a previous candidate's artifacts to a later
    /// [`Self::extract_from_dir`] (the harness re-points the link only after
    /// a run completes).
    pub fn run_profile_in(&self, work_dir: &Path) -> Result<(), MeasurementError> {
        self.run_profile_in_with(work_dir, self.needs_allocative())
    }

    /// [`run_profile_in`](Self::run_profile_in) with the allocative lane
    /// explicitly on or off. `optimize` shares one profile run across every
    /// objective of a workload, so it ORs their
    /// [`needs_allocative`](Self::needs_allocative) — a time metric's value
    /// is unaffected by the lane beyond its (noise-level) snapshot cost.
    pub fn run_profile_in_with(
        &self,
        work_dir: &Path,
        allocative: bool,
    ) -> Result<(), MeasurementError> {
        let stale = work_dir.join(format!("benchmark-runs/latest_{}", self.trace_name()));
        if let Err(e) = std::fs::remove_file(&stale) {
            if e.kind() != std::io::ErrorKind::NotFound {
                return Err(MeasurementError::new(format!(
                    "removing stale latest link {}: {e}",
                    stale.display()
                )));
            }
        }
        let scale = self.scale().to_string();
        let features = if allocative {
            "profiling,allocative"
        } else {
            "profiling"
        };
        let status = Command::new("cargo")
            .current_dir(work_dir)
            .args([
                "run",
                "--release",
                "-p",
                "jolt-prover",
                "--features",
                features,
                "--bin",
                "jolt-prover",
                "--",
                "profile",
                "--name",
                self.workload,
                "--scale",
                &scale,
                "--format",
                "chrome",
            ])
            .status()
            .map_err(|e| MeasurementError::new(format!("spawning profile bin: {e}")))?;
        if !status.success() {
            return Err(MeasurementError::new(format!(
                "profile bin failed with {status} for {}",
                self.key
            )));
        }
        Ok(())
    }

    /// Reads the metric from the summary a prior [`Self::run_profile_in`]
    /// left under `work_dir`.
    pub fn extract_from_dir(&self, work_dir: &Path) -> Result<f64, MeasurementError> {
        let path = self.summary_path(work_dir);
        let data = std::fs::read_to_string(&path)
            .map_err(|e| MeasurementError::new(format!("reading {}: {e}", path.display())))?;
        let summary: Value = serde_json::from_str(&data)
            .map_err(|e| MeasurementError::new(format!("parsing {}: {e}", path.display())))?;
        self.extract(&summary)
    }

    /// One-shot measurement: profile run + metric extraction.
    pub fn measure_in(&self, work_dir: &Path) -> Result<f64, MeasurementError> {
        self.run_profile_in(work_dir)?;
        self.extract_from_dir(work_dir)
    }

    /// Reads the metric out of a parsed `summary.json`. An absent label or
    /// null field is a measurement error, never `0.0`.
    pub fn extract(&self, summary: &Value) -> Result<f64, MeasurementError> {
        let schema_version = summary
            .get("schema_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| MeasurementError::new("summary missing schema_version"))?;
        if schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(MeasurementError::new(format!(
                "unsupported summary schema_version {schema_version} (expected {SUPPORTED_SCHEMA_VERSION})"
            )));
        }
        let ns_to_s = |ns: f64| ns / 1e9;
        match self.metric {
            TelemetryMetric::ProverTimeS => summary
                .pointer("/root/wall_time_ns")
                .and_then(Value::as_f64)
                .map(ns_to_s)
                .ok_or_else(|| MeasurementError::new("summary has no root span")),
            TelemetryMetric::PeakRssGib => summary
                .get("peak_rss_gib")
                .and_then(Value::as_f64)
                .ok_or_else(|| MeasurementError::new("summary has no peak_rss_gib")),
            TelemetryMetric::PeakMemoryGib => summary
                .pointer("/root/peak_memory_gib")
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    MeasurementError::new(
                        "summary has no root peak_memory_gib (no monitor samples in the root span)",
                    )
                }),
            TelemetryMetric::Total(label) | TelemetryMetric::SelfTime(label) => {
                let field = match self.metric {
                    TelemetryMetric::Total(_) => "total_ns",
                    _ => "self_ns",
                };
                summary
                    .get("spans")
                    .and_then(|spans| spans.get(label))
                    .and_then(|span| span.get(field))
                    .and_then(Value::as_f64)
                    .map(ns_to_s)
                    .ok_or_else(|| {
                        MeasurementError::new(format!(
                            "span label {label:?} absent from summary (key {}); \
                             absent labels are an error, never 0.0",
                            self.key
                        ))
                    })
            }
            TelemetryMetric::Heap { snapshot, root } => {
                let entry = summary
                    .get("heap")
                    .and_then(|heap| heap.get(snapshot))
                    .ok_or_else(|| {
                        MeasurementError::new(format!(
                            "heap snapshot {snapshot:?} absent from summary (key {}); the \
                             profile run must include the allocative lane, and absent \
                             snapshots are an error, never 0.0",
                            self.key
                        ))
                    })?;
                match root {
                    None => entry
                        .get("total_bytes")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| {
                            MeasurementError::new(format!(
                                "heap snapshot {snapshot:?} has no total_bytes (key {})",
                                self.key
                            ))
                        }),
                    Some(root) => entry
                        .get("roots")
                        .and_then(|roots| roots.get(root))
                        .and_then(Value::as_f64)
                        .ok_or_else(|| {
                            MeasurementError::new(format!(
                                "root frame {root:?} absent from heap snapshot {snapshot:?} \
                                 (key {}); absent roots are an error, never 0.0",
                                self.key
                            ))
                        }),
                }
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_metric() {
        let obj = TelemetryObjective::parse("telemetry:fibonacci:prover_time_s").unwrap();
        assert_eq!(obj.workload, "fibonacci");
        assert_eq!(obj.metric, TelemetryMetric::ProverTimeS);
        assert_eq!(obj.scale(), 16);
        assert_eq!(obj.name(), "telemetry:fibonacci:prover_time_s");
    }

    #[test]
    fn label_after_third_colon_is_verbatim() {
        let obj =
            TelemetryObjective::parse("telemetry:sha2-chain:total:EqPolynomial::evals").unwrap();
        assert_eq!(obj.metric, TelemetryMetric::Total("EqPolynomial::evals"));
        assert_eq!(obj.scale(), 22);
        let selfed =
            TelemetryObjective::parse("telemetry:sha2-chain:self:Stage1Batch::prove").unwrap();
        assert_eq!(
            selfed.metric,
            TelemetryMetric::SelfTime("Stage1Batch::prove")
        );
    }

    #[test]
    fn rejects_unknown_workload_and_metric() {
        assert!(TelemetryObjective::parse("telemetry:nope:prover_time_s").is_err());
        assert!(TelemetryObjective::parse("telemetry:fibonacci:bogus").is_err());
        assert!(TelemetryObjective::parse("telemetry:fibonacci").is_err());
        assert!(TelemetryObjective::parse("telemetry:fibonacci:heap:").is_err());
        assert!(TelemetryObjective::parse("callgrind:x:instructions").is_err());
    }

    #[test]
    fn parses_heap_metrics_with_verbatim_roots() {
        let total =
            TelemetryObjective::parse("telemetry:fibonacci:heap:Stage2Batch_prepared").unwrap();
        assert_eq!(
            total.metric,
            TelemetryMetric::Heap {
                snapshot: "Stage2Batch_prepared",
                root: None
            }
        );
        assert!(total.needs_allocative());
        assert_eq!(total.units(), Some("bytes"));

        // The root frame is verbatim after the snapshot's colon — kernel
        // type names contain `::` and generics.
        let root = TelemetryObjective::parse(
            "telemetry:fibonacci:heap:Stage2Batch_prepared:NaiveSumcheckProver<Fr, RamReadWriteChecking<Fr>>",
        )
        .unwrap();
        assert_eq!(
            root.metric,
            TelemetryMetric::Heap {
                snapshot: "Stage2Batch_prepared",
                root: Some("NaiveSumcheckProver<Fr, RamReadWriteChecking<Fr>>")
            }
        );

        let time = TelemetryObjective::parse("telemetry:fibonacci:prover_time_s").unwrap();
        assert!(!time.needs_allocative());
    }

    #[test]
    fn heap_extraction_reads_totals_and_roots_and_errors_on_absence() {
        let summary: Value = serde_json::json!({
            "schema_version": 1,
            "heap": {
                "Stage2Batch_prepared": {
                    "total_bytes": 8_596_875_264u64,
                    "roots": {
                        "KernelA": 8_594_128_896u64,
                        "ProofSession": 48u64,
                    }
                }
            }
        });
        let total =
            TelemetryObjective::parse("telemetry:fibonacci:heap:Stage2Batch_prepared").unwrap();
        assert_eq!(total.extract(&summary).unwrap(), 8_596_875_264.0);
        let root =
            TelemetryObjective::parse("telemetry:fibonacci:heap:Stage2Batch_prepared:KernelA")
                .unwrap();
        assert_eq!(root.extract(&summary).unwrap(), 8_594_128_896.0);

        let absent_snapshot =
            TelemetryObjective::parse("telemetry:fibonacci:heap:Stage9_prepared").unwrap();
        assert!(absent_snapshot.extract(&summary).is_err());
        let absent_root =
            TelemetryObjective::parse("telemetry:fibonacci:heap:Stage2Batch_prepared:KernelB")
                .unwrap();
        assert!(absent_root.extract(&summary).is_err());
        // An allocative-less run serializes "heap": {} — still an error.
        let lane_off: Value = serde_json::json!({ "schema_version": 1, "heap": {} });
        assert!(total.extract(&lane_off).is_err());
    }

    #[test]
    fn absent_label_is_an_error_never_zero() {
        let summary: Value = serde_json::json!({
            "schema_version": 1,
            "root": { "wall_time_ns": 2_000_000_000u64, "peak_memory_gib": null },
            "peak_rss_gib": 4.0,
            "spans": { "prove_batch": { "count": 8, "total_ns": 1_500_000_000u64, "self_ns": 700_000_000u64 } },
        });
        let time = TelemetryObjective::parse("telemetry:fibonacci:prover_time_s").unwrap();
        assert_eq!(time.extract(&summary).unwrap(), 2.0);
        let total = TelemetryObjective::parse("telemetry:fibonacci:total:prove_batch").unwrap();
        assert_eq!(total.extract(&summary).unwrap(), 1.5);
        let selfed = TelemetryObjective::parse("telemetry:fibonacci:self:prove_batch").unwrap();
        assert_eq!(selfed.extract(&summary).unwrap(), 0.7);

        let absent = TelemetryObjective::parse("telemetry:fibonacci:total:NoSuchSpan").unwrap();
        assert!(absent.extract(&summary).is_err());
        let peak_mem = TelemetryObjective::parse("telemetry:fibonacci:peak_memory_gib").unwrap();
        assert!(
            peak_mem.extract(&summary).is_err(),
            "null is an error, not 0.0"
        );
        let rss = TelemetryObjective::parse("telemetry:fibonacci:peak_rss_gib").unwrap();
        assert_eq!(rss.extract(&summary).unwrap(), 4.0);
    }

    #[test]
    fn rejects_future_schema_versions() {
        let summary = serde_json::json!({ "schema_version": 2 });
        let obj = TelemetryObjective::parse("telemetry:fibonacci:prover_time_s").unwrap();
        assert!(obj.extract(&summary).is_err());
    }

    /// The curated consts and the runtime parser must agree — a HashMap
    /// keyed by the parsed objective must hit the const-keyed entry.
    #[test]
    fn curated_consts_round_trip_through_parser() {
        let curated = [
            MODULAR_PROVER_TIME,
            MODULAR_COMMIT_TIME,
            MODULAR_ROUND_LOOP_TIME,
        ]
        .into_iter()
        .chain(MODULAR_STAGE_TOTALS);
        for objective in curated {
            let parsed = OptimizationObjective::from_key(objective.name())
                .expect("curated key inside grammar")
                .expect("curated key parses");
            assert!(
                parsed == objective,
                "parse mismatch for {}",
                objective.name()
            );
        }
    }

    #[test]
    fn summary_path_maps_hyphens() {
        let obj = TelemetryObjective::parse("telemetry:sha2-chain:prover_time_s").unwrap();
        assert_eq!(
            obj.summary_path(Path::new("/work")),
            Path::new("/work/benchmark-runs/latest_modular_sha2_chain_22/summary.json")
        );
    }
}
