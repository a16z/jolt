//! Deterministic instruction-count objectives over iai-callgrind
//! microbenchmarks (`callgrind:<bench-name>:instructions`).

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use super::MeasurementError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallgrindObjective {
    pub key: &'static str,
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
        if bench.is_empty() {
            return Err(MeasurementError::new(format!(
                "empty callgrind bench name in {key}"
            )));
        }
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
            "iai-callgrind instruction count of bench {} (requires Valgrind)",
            self.bench
        )
    }

    pub fn units(&self) -> Option<&str> {
        Some("Ir")
    }

    pub fn measure_in(&self, work_dir: &Path) -> Result<f64, MeasurementError> {
        if Command::new("valgrind").arg("--version").output().is_err() {
            return Err(MeasurementError::new(
                "valgrind not found: install Valgrind and iai-callgrind-runner matching the workspace iai-callgrind version",
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

const SUPPORTED_BENCHMARK_SUMMARY_VERSION: &str = "6";

#[derive(Deserialize)]
struct BenchmarkSummary {
    version: String,
    profiles: Vec<Profile>,
}

#[derive(Deserialize)]
struct Profile {
    tool: String,
    summaries: ProfileData,
}

#[derive(Deserialize)]
struct ProfileData {
    total: ProfileTotal,
}

#[derive(Deserialize)]
struct ProfileTotal {
    summary: serde_json::Value,
}

impl ProfileTotal {
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

#[derive(Deserialize)]
struct MetricsDiff {
    metrics: EitherOrBoth,
}

#[derive(Deserialize)]
enum EitherOrBoth {
    Both(Metric, #[expect(dead_code, reason = "wire shape")] Metric),
    Left(Metric),
    Right(#[expect(dead_code, reason = "wire shape")] Metric),
}

#[derive(Deserialize)]
enum Metric {
    Int(u64),
    Float(f64),
}

impl Metric {
    fn as_f64(&self) -> f64 {
        match *self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

impl EitherOrBoth {
    fn new_value(&self) -> Option<f64> {
        match self {
            Self::Both(new, _) | Self::Left(new) => Some(new.as_f64()),
            Self::Right(_) => None,
        }
    }
}

fn parse_instruction_count(stdout: &str) -> Result<f64, MeasurementError> {
    let mut total = 0.0;
    let mut found = false;
    for document in serde_json::Deserializer::from_str(stdout).into_iter::<BenchmarkSummary>() {
        let document = document.map_err(|e| {
            MeasurementError::new(format!("malformed iai-callgrind JSON output: {e}"))
        })?;
        if document.version != SUPPORTED_BENCHMARK_SUMMARY_VERSION {
            return Err(MeasurementError::new(format!(
                "unsupported iai-callgrind summary version {:?} (expected {:?})",
                document.version, SUPPORTED_BENCHMARK_SUMMARY_VERSION
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
    found
        .then_some(total)
        .ok_or_else(|| MeasurementError::new("no Callgrind Ir total in iai-callgrind JSON output"))
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn fixture_document(ir: u64, old_ir: Option<u64>) -> String {
        let metrics = match old_ir {
            None => serde_json::json!({ "Left": { "Int": ir } }),
            Some(old) => serde_json::json!({ "Both": [{ "Int": ir }, { "Int": old }] }),
        };
        serde_json::json!({
            "version": "6",
            "profiles": [{
                "tool": "Callgrind",
                "summaries": {
                    "total": {
                        "summary": { "Callgrind": { "Ir": { "metrics": metrics } } }
                    }
                }
            }]
        })
        .to_string()
    }

    #[test]
    fn parses_key() {
        let objective = CallgrindObjective::parse("callgrind:eq_evals:instructions").unwrap();
        assert_eq!(objective.bench, "eq_evals");
        assert_eq!(objective.name(), "callgrind:eq_evals:instructions");
    }

    #[test]
    fn rejects_malformed_keys() {
        assert!(CallgrindObjective::parse("callgrind:eq_evals").is_err());
        assert!(CallgrindObjective::parse("callgrind::instructions").is_err());
        assert!(CallgrindObjective::parse("callgrind:eq_evals:cycles").is_err());
        assert!(CallgrindObjective::parse("telemetry:x:y").is_err());
    }

    #[test]
    fn extracts_fresh_and_baselined_ir() {
        assert_eq!(
            parse_instruction_count(&fixture_document(123_456, None)).unwrap(),
            123_456.0
        );
        assert_eq!(
            parse_instruction_count(&fixture_document(1000, Some(2500))).unwrap(),
            1000.0
        );
    }

    #[test]
    fn sums_cases_and_rejects_invalid_output() {
        let output = format!(
            "{}\n{}\n",
            fixture_document(1000, None),
            fixture_document(2500, Some(2400))
        );
        assert_eq!(parse_instruction_count(&output).unwrap(), 3500.0);
        assert!(parse_instruction_count("").is_err());
        assert!(parse_instruction_count("not json").is_err());
        assert!(parse_instruction_count(
            &fixture_document(1, None).replace(r#""version":"6""#, r#""version":"7""#)
        )
        .is_err());
    }
}
