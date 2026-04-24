pub mod binding;
pub mod jolt_crypto_g1_msm;
pub mod jolt_crypto_g1_scalar_mul;
pub mod jolt_crypto_gt_scalar_mul;
pub mod jolt_crypto_pedersen_commit;
pub mod naive_sort;
pub mod prover_time;

use std::path::Path;

/// Read the point estimate (mean, in seconds) from Criterion's output
/// for a given benchmark and baseline name.
///
/// `work_dir` is the directory where `cargo bench` was invoked — Criterion
/// writes its output under `{work_dir}/target/criterion/`.
pub fn read_criterion_estimate(work_dir: &Path, bench_name: &str, baseline: &str) -> Option<f64> {
    let path = work_dir
        .join("target/criterion")
        .join(bench_name)
        .join(baseline)
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
