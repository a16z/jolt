pub mod binding;
pub mod field_mul;
pub mod naive_sort;
pub mod prover_time;
pub mod trace_gen;

use std::path::Path;

/// Read the point estimate (mean, in seconds) from Criterion's output
/// for a given benchmark and baseline name.
///
/// `work_dir` is the workspace root where `cargo bench -p jolt-eval` was
/// invoked. Criterion writes to `CRITERION_HOME`, else
/// `$CARGO_TARGET_DIR/criterion`, else `target/criterion`. Cargo runs benchmarks
/// from the package root, so relative overrides are resolved against
/// `{work_dir}/jolt-eval`. A target directory set only in Cargo configuration
/// is not detected.
pub fn read_criterion_estimate(work_dir: &Path, bench_name: &str, baseline: &str) -> Option<f64> {
    let package_dir = work_dir.join("jolt-eval");
    let criterion_dir = if let Some(home) = std::env::var_os("CRITERION_HOME") {
        package_dir.join(home)
    } else if let Some(target) = std::env::var_os("CARGO_TARGET_DIR") {
        package_dir.join(target).join("criterion")
    } else {
        work_dir.join("target/criterion")
    };
    let path = criterion_dir
        .join(bench_name)
        .join(baseline)
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
