//! Deterministic instruction-count objectives over iai-callgrind
//! microbenchmarks (`callgrind:<bench-name>:instructions`).
//!
//! The noise-free signal for optimizer accept/reject decisions: callgrind's
//! `Ir` event kind (instruction count, rendered as "Instructions" only in
//! console output), parsed from iai-callgrind's machine-readable JSON output
//! (`--output-format=json`). Opt-in: requires Valgrind and the
//! `iai-callgrind-runner`; both absent produce a clear measurement error,
//! never a silent zero. Bench targets live in `jolt-eval/benches/callgrind/`
//! (explicit `path`/`harness = false` entries that `sync_targets.sh`
//! preserves).

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use super::MeasurementError;

/// One parsed `callgrind:<bench-name>:instructions` objective.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallgrindObjective {
    /// The full verbatim key (also the objective's CLI name).
    pub key: &'static str,
    /// The cargo bench target under `jolt-eval/benches/callgrind/`.
    pub bench: &'static str,
}

impl CallgrindObjective {
    pub fn parse(key: &str) -> Result<Self, MeasurementError> {
        let rest = key
            .strip_prefix("callgrind:")
            .ok_or_else(|| MeasurementError::new(format!("not a callgrind key: {key}")))?;
        let (bench, metric) = rest.split_once(':').ok_or_else(|| {
            MeasurementError::new(format!(
                "malformed callgrind key {key}: expected callgrind:<bench-name>:instructions"
            ))
        })?;
        if metric != "instructions" {
            return Err(MeasurementError::new(format!(
                "unknown callgrind metric {metric:?} in {key}: only \"instructions\" is supported"
            )));
        }
        Ok(Self {
            key: Box::leak(key.to_string().into_boxed_str()),
            bench: Box::leak(bench.to_string().into_boxed_str()),
        })
    }

    pub fn name(&self) -> &str {
        self.key
    }

    pub fn description(&self) -> String {
        format!(
            "iai-callgrind instruction count of bench {} (deterministic; requires Valgrind)",
            self.bench
        )
    }

    pub fn units(&self) -> Option<&str> {
        Some("Ir")
    }

    /// Runs the bench under callgrind in `work_dir` and sums the `Ir` totals
    /// across its benchmark cases.
    pub fn measure_in(&self, work_dir: &Path) -> Result<f64, MeasurementError> {
        if Command::new("valgrind").arg("--version").output().is_err() {
            return Err(MeasurementError::new(
                "valgrind not found: the callgrind lane is opt-in and needs Valgrind plus \
                 `cargo install iai-callgrind-runner` (matching the workspace iai-callgrind \
                 version)",
            ));
        }
        let output = Command::new("cargo")
            .current_dir(work_dir)
            .args([
                "bench",
                "-p",
                "jolt-eval",
                "--bench",
                self.bench,
                "--",
                "--output-format=json",
            ])
            .output()
            .map_err(|e| MeasurementError::new(format!("spawning cargo bench: {e}")))?;
        if !output.status.success() {
            return Err(MeasurementError::new(format!(
                "callgrind bench {} failed with {}: {}",
                self.bench,
                output.status,
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or_default()
            )));
        }
        parse_instruction_count(&String::from_utf8_lossy(&output.stdout))
    }
}

/// The `BenchmarkSummary.version` this parser understands. iai-callgrind
/// bumps it only on backwards-incompatible changes to the summary format
/// (`SCHEMA_VERSION` in `iai-callgrind-runner/src/runner/summary.rs`;
/// 0.16.x emits "6"), so an unknown version is a loud error rather than a
/// silently misparsed measurement.
const SUPPORTED_BENCHMARK_SUMMARY_VERSION: &str = "6";

/// Minimal structural mirror of iai-callgrind's `BenchmarkSummary` JSON —
/// only the spine the `Ir` extraction walks. With `--output-format=json` the
/// runner serializes one `BenchmarkSummary` document per benchmark case to
/// stdout (compact, newline-separated). The `Ir` total for a case lives at
///
/// `profiles[tool == "Callgrind"].summaries.total.summary.Callgrind.Ir.metrics`
///
/// as a `MetricsDiff` whose `metrics` is `EitherOrBoth<Metric>`: `Left(new)`
/// on a fresh run, `Both(new, old)` when a baseline exists (left is `new`
/// per the runner's convention), `Right(old)` only when the new run produced
/// no metric. Unlisted fields are ignored by serde's default, so additive
/// runner changes don't break parsing; the `version` gate catches the
/// incompatible ones.
#[derive(Deserialize)]
struct BenchmarkSummary {
    version: String,
    profiles: Vec<Profile>,
}

/// One valgrind tool's run within a benchmark case (`Profile` in the runner).
#[derive(Deserialize)]
struct Profile {
    /// The tool name (`"Callgrind"`, `"DHAT"`, …).
    tool: String,
    summaries: ProfileData,
}

/// The tool run's parts and their total (`ProfileData` in the runner). Only
/// the total is read — it always exists and already sums multi-part runs
/// (subprocesses/threads under `--trace-children`), so reading parts too
/// would double-count.
#[derive(Deserialize)]
struct ProfileData {
    total: ProfileTotal,
}

#[derive(Deserialize)]
struct ProfileTotal {
    /// The runner's `ToolMetricSummary`, an externally tagged per-tool enum
    /// (`{"Callgrind": {event kind → MetricsDiff}}`, `"None"`, …). Kept as a
    /// raw value and drilled with the explicit `Callgrind`/`Ir` path so
    /// tools this parser never reads — including ones added in future
    /// runner versions — can't break deserialization.
    summary: serde_json::Value,
}

impl ProfileTotal {
    /// The new-run `Ir` total, if this tool summary is Callgrind's and the
    /// new run produced one.
    fn new_ir(&self) -> Result<Option<f64>, MeasurementError> {
        let Some(ir) = self.summary.pointer("/Callgrind/Ir") else {
            return Ok(None);
        };
        let diff: MetricsDiff = serde_json::from_value(ir.clone()).map_err(|e| {
            MeasurementError::new(format!(
                "malformed Ir MetricsDiff in callgrind summary: {e}"
            ))
        })?;
        Ok(diff.metrics.new_value())
    }
}

/// New/old metric pair for one event kind (`MetricsDiff` in the runner).
#[derive(Deserialize)]
struct MetricsDiff {
    metrics: EitherOrBoth,
}

/// The runner's `EitherOrBoth<Metric>`: left is `new`, right is `old`. The
/// old-value slots exist only to match the wire shape — the measurement is
/// always the new run.
#[derive(Deserialize)]
enum EitherOrBoth {
    Both(Metric, #[expect(dead_code, reason = "wire shape")] Metric),
    Left(Metric),
    Right(#[expect(dead_code, reason = "wire shape")] Metric),
}

/// The runner's `Metric`: valgrind counts are `u64`, derived metrics `f64`.
#[derive(Deserialize)]
enum Metric {
    Int(u64),
    Float(f64),
}

impl Metric {
    fn as_f64(&self) -> f64 {
        match *self {
            Self::Int(n) => n as f64,
            Self::Float(f) => f,
        }
    }
}

impl EitherOrBoth {
    /// The new run's value, if the new run produced one.
    fn new_value(&self) -> Option<f64> {
        match self {
            Self::Both(new, _) | Self::Left(new) => Some(new.as_f64()),
            Self::Right(_) => None,
        }
    }
}

/// Sums the new-run `Ir` totals across the benchmark-case documents on
/// stdout. Malformed documents, an unsupported summary version, and output
/// without a single Callgrind `Ir` total are all loud errors — a partial or
/// silently-zero instruction count would corrupt optimizer decisions.
fn parse_instruction_count(stdout: &str) -> Result<f64, MeasurementError> {
    let mut total = 0.0;
    let mut found = false;
    for document in serde_json::Deserializer::from_str(stdout).into_iter::<BenchmarkSummary>() {
        let document = document.map_err(|e| {
            MeasurementError::new(format!("malformed iai-callgrind JSON output: {e}"))
        })?;
        if document.version != SUPPORTED_BENCHMARK_SUMMARY_VERSION {
            return Err(MeasurementError::new(format!(
                "unsupported iai-callgrind summary version {:?} (expected \
                 {SUPPORTED_BENCHMARK_SUMMARY_VERSION:?}); update the parser against the runner's \
                 summary schema",
                document.version
            )));
        }
        for profile in &document.profiles {
            if profile.tool != "Callgrind" {
                continue;
            }
            if let Some(ir) = profile.summaries.total.new_ir()? {
                total += ir;
                found = true;
            }
        }
    }
    if found {
        Ok(total)
    } else {
        Err(MeasurementError::new(
            "no Callgrind Ir total in iai-callgrind JSON output",
        ))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_key() {
        let obj = CallgrindObjective::parse("callgrind:eq_evals:instructions").unwrap();
        assert_eq!(obj.bench, "eq_evals");
        assert_eq!(obj.name(), "callgrind:eq_evals:instructions");
    }

    #[test]
    fn rejects_malformed_keys() {
        assert!(CallgrindObjective::parse("callgrind:eq_evals").is_err());
        assert!(CallgrindObjective::parse("callgrind:eq_evals:cycles").is_err());
        assert!(CallgrindObjective::parse("telemetry:x:y").is_err());
    }

    /// One benchmark-case document shaped exactly like iai-callgrind 0.16.1
    /// runner output (`BenchmarkSummary`, schema version "6"): full metric
    /// map with `MetricsDiff` wrappers, part details, flanking tool fields.
    /// The `real_runner_types_accept_the_fixture` test deserializes it
    /// through the actual runner structs, so this fixture cannot drift from
    /// the real schema.
    fn fixture_document(ir: u64, old_ir: Option<u64>) -> String {
        let metrics = match old_ir {
            None => format!(r#"{{"Left":{{"Int":{ir}}}}}"#),
            Some(old) => format!(r#"{{"Both":[{{"Int":{ir}}},{{"Int":{old}}}]}}"#),
        };
        let diffs = if old_ir.is_some() {
            r#"{"diff_pct":"0","factor":"1"}"#
        } else {
            "null"
        };
        format!(
            r#"{{
              "version": "6",
              "kind": "LibraryBenchmark",
              "function_name": "eq_evals",
              "module_path": "eq_evals::bench_group::eq_evals",
              "id": "small",
              "details": null,
              "benchmark_exe": "/target/release/deps/eq_evals-1234",
              "benchmark_file": "/jolt-eval/benches/callgrind/eq_evals.rs",
              "package_dir": "/jolt-eval",
              "project_root": "/",
              "baselines": [null, null],
              "summary_output": null,
              "profiles": [
                {{
                  "tool": "Callgrind",
                  "log_paths": ["/target/iai/callgrind.log"],
                  "out_paths": ["/target/iai/callgrind.out"],
                  "flamegraphs": [],
                  "summaries": {{
                    "parts": [
                      {{
                        "details": {{
                          "Left": {{
                            "command": "/target/release/deps/eq_evals-1234",
                            "pid": 12345,
                            "parent_pid": null,
                            "part": 1,
                            "thread": 1,
                            "details": null,
                            "path": "/target/iai/callgrind.out"
                          }}
                        }},
                        "metrics_summary": {{
                          "Callgrind": {{
                            "Ir": {{"diffs": {diffs}, "metrics": {metrics}}}
                          }}
                        }}
                      }}
                    ],
                    "total": {{
                      "regressions": [],
                      "summary": {{
                        "Callgrind": {{
                          "Ir": {{"diffs": {diffs}, "metrics": {metrics}}},
                          "Dr": {{"diffs": null, "metrics": {{"Left": {{"Int": 7}}}}}},
                          "EstimatedCycles": {{"diffs": null, "metrics": {{"Left": {{"Float": 42.5}}}}}}
                        }}
                      }}
                    }}
                  }}
                }}
              ]
            }}"#
        )
    }

    /// The fixture must deserialize through the real
    /// `iai-callgrind-runner` summary structs — the schema this parser
    /// mirrors. A runner upgrade that reshapes the summary fails here first.
    #[test]
    fn real_runner_types_accept_the_fixture() {
        use iai_callgrind_runner::runner::summary::BenchmarkSummary as RealSummary;
        for fixture in [
            fixture_document(1000, None),
            fixture_document(1000, Some(900)),
        ] {
            let parsed: RealSummary = serde_json::from_str(&fixture).unwrap();
            // And our extraction agrees with the runner's own reserialization.
            let reserialized = serde_json::to_string(&parsed).unwrap();
            assert_eq!(parse_instruction_count(&reserialized).unwrap(), 1000.0);
        }
    }

    #[test]
    fn extracts_ir_from_fresh_run_metrics() {
        assert_eq!(
            parse_instruction_count(&fixture_document(123_456, None)).unwrap(),
            123_456.0
        );
    }

    #[test]
    fn extracts_new_ir_when_a_baseline_is_present() {
        // `Both(new, old)`: the measurement is the new run, never the old.
        assert_eq!(
            parse_instruction_count(&fixture_document(1000, Some(2500))).unwrap(),
            1000.0
        );
    }

    #[test]
    fn sums_ir_across_case_documents() {
        let stdout = format!(
            "{}\n{}\n",
            fixture_document(1000, None),
            fixture_document(2500, Some(2400))
        );
        assert_eq!(parse_instruction_count(&stdout).unwrap(), 3500.0);
    }

    #[test]
    fn missing_ir_is_an_error() {
        // A document without any Callgrind profile carries no Ir.
        let no_callgrind = fixture_document(1, None).replace("\"Callgrind\"", "\"Cachegrind\"");
        assert!(parse_instruction_count(&no_callgrind).is_err());
        assert!(parse_instruction_count("").is_err());
    }

    #[test]
    fn unsupported_summary_version_is_an_error() {
        let v7 = fixture_document(1000, None).replace(r#""version": "6""#, r#""version": "7""#);
        let err = parse_instruction_count(&v7).unwrap_err();
        assert!(err.to_string().contains("unsupported"), "{err}");
    }

    #[test]
    fn malformed_output_is_an_error() {
        assert!(parse_instruction_count(r#"{"kind":"benchmark"}"#).is_err());
        assert!(parse_instruction_count("not json").is_err());
    }
}
