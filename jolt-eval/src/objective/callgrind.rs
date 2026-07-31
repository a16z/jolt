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

use serde_json::Value;

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

/// Sums the `Ir` instruction counts from iai-callgrind's JSON output (one
/// JSON document per benchmark case on stdout).
fn parse_instruction_count(stdout: &str) -> Result<f64, MeasurementError> {
    let mut total = 0.0;
    let mut found = false;
    let mut deserializer = serde_json::Deserializer::from_str(stdout).into_iter::<Value>();
    for document in deserializer.by_ref() {
        let Ok(document) = document else { break };
        if let Some(ir) = find_ir(&document) {
            total += ir;
            found = true;
        }
    }
    if found {
        Ok(total)
    } else {
        Err(MeasurementError::new(
            "no Ir event kind in iai-callgrind JSON output",
        ))
    }
}

/// Depth-first search for the first `Ir` metric in one benchmark summary
/// document. iai-callgrind nests the metrics under the profile's total
/// summary; the first `Ir` in document order is that total.
fn find_ir(value: &Value) -> Option<f64> {
    match value {
        Value::Object(map) => {
            if let Some(ir) = map.get("Ir") {
                if let Some(ir) = as_metric_number(ir) {
                    return Some(ir);
                }
            }
            map.values().find_map(find_ir)
        }
        Value::Array(items) => items.iter().find_map(find_ir),
        _ => None,
    }
}

/// iai-callgrind encodes metric values either as bare numbers or as
/// single-variant wrappers like `{"Int": 12345}`.
fn as_metric_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Object(map) if map.len() == 1 => map.values().next().and_then(as_metric_number),
        _ => None,
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

    #[test]
    fn sums_ir_across_documents() {
        let stdout = r#"
            {"kind":"benchmark","callgrind_summary":{"summaries":[{"events":{"Ir":{"Int":1000}}}]}}
            {"kind":"benchmark","callgrind_summary":{"summaries":[{"events":{"Ir":2500}}]}}
        "#;
        assert_eq!(parse_instruction_count(stdout).unwrap(), 3500.0);
    }

    #[test]
    fn missing_ir_is_an_error() {
        assert!(parse_instruction_count(r#"{"kind":"benchmark"}"#).is_err());
        assert!(parse_instruction_count("").is_err());
    }
}
