pub mod binding;
pub mod prover_time;

use std::path::Path;

/// Read the point estimate (mean, in seconds) from Criterion's output
/// for a given benchmark and baseline name.
pub fn read_criterion_estimate(bench_name: &str, baseline: &str) -> Option<f64> {
    let path = Path::new("target/criterion")
        .join(bench_name)
        .join(baseline)
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
