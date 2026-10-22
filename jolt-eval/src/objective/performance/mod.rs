pub mod binding;
pub mod field_mul;
pub mod naive_sort;
pub mod prover_time;
pub mod source_trace_gen;
pub mod trace_gen;

use serde_json::Value;
use source_trace_gen::PROGRAMS;
use std::path::Path;

/// Reads a Criterion mean in seconds for a benchmark and baseline.
/// `source_trace_gen` instead sums the three source-generation medians;
/// reference generation and scanning are comparison measurements.
///
/// `work_dir` is the directory where `cargo bench` was invoked — Criterion
/// writes its output under `{work_dir}/target/criterion/`.
pub fn read_criterion_estimate(work_dir: &Path, bench_name: &str, baseline: &str) -> Option<f64> {
    if bench_name == "source_trace_gen" {
        return PROGRAMS.iter().try_fold(0.0, |total, (label, _, _)| {
            // Criterion replaces the slash in each group name with an underscore.
            let path = work_dir
                .join("target/criterion")
                .join(format!("source_trace_gen_{label}"))
                .join("source")
                .join(baseline)
                .join("estimates.json");
            let data = std::fs::read_to_string(path).ok()?;
            let json: Value = serde_json::from_str(&data).ok()?;
            let nanos = json.get("median")?.get("point_estimate")?.as_f64()?;
            Some(total + nanos / 1e9)
        });
    }
    let path = work_dir
        .join("target/criterion")
        .join(bench_name)
        .join(baseline)
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}

#[cfg(test)]
mod tests {
    use super::read_criterion_estimate;

    #[test]
    fn source_trace_objective_sums_source_medians() {
        let dir = tempfile::tempdir().unwrap();
        for (label, nanos) in [
            ("alu", 2_000_000),
            ("memory", 3_000_000),
            ("call_frame", 5_000_000),
        ] {
            let path = dir.path().join(format!(
                "target/criterion/source_trace_gen_{label}/source/new"
            ));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("estimates.json"),
                format!(
                    r#"{{"median":{{"point_estimate":{nanos}}},"mean":{{"point_estimate":999}}}}"#
                ),
            )
            .unwrap();
        }
        assert_eq!(
            read_criterion_estimate(dir.path(), "source_trace_gen", "new"),
            Some(0.01)
        );
        assert_eq!(
            read_criterion_estimate(dir.path(), "source_trace_gen", "missing"),
            None
        );
    }
}
