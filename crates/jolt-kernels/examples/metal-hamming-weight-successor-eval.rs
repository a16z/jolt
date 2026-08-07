#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error, time::Instant};

use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::hamming_weight_claim_reduction::{
    execute_hamming_weight_claim_reduction_fixture, HammingWeightResidentRow, HammingWeightShape,
    HammingWeightSuccessorConfig, HAMMING_WEIGHT_TARGET_EIGHT_X_NS,
    HAMMING_WEIGHT_TARGET_FIVE_X_NS, HAMMING_WEIGHT_TARGET_GPU_ACTIVE_NS,
};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let log_t = env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(26);
    if !(18..=28).contains(&log_t) {
        return Err("log_t must be in 18..=28".into());
    }
    let row_count = 1usize << log_t;
    let row_build_started = Instant::now();
    let rows = (0..row_count)
        .into_par_iter()
        .map(fixture_row)
        .collect::<Vec<_>>();
    let row_build_wall = row_build_started.elapsed();

    let point = (0..log_t)
        .map(|coordinate| AkitaField::from_u64((coordinate * 17 + 3) as u64))
        .collect::<Vec<_>>();
    let config = HammingWeightSuccessorConfig::default();
    let shape = HammingWeightShape::new(row_count, config)?;
    let outer_log2 = log_t - shape.inner_log2();
    let (outer_point, inner_point) = point.split_at(outer_log2);
    let e_out = EqPolynomial::evals(outer_point, None);
    let e_in = EqPolynomial::evals(inner_point, None);

    let execution_started = Instant::now();
    let execution = execute_hamming_weight_claim_reduction_fixture(&rows, &e_in, &e_out, config)?;
    let execution_wall = execution_started.elapsed();
    let gpu_active_ns = execution.gpu_active.as_nanos() as u64;
    let target_scale = log_t == 26;

    println!(
        "{}",
        json!({
            "schema_version": 1,
            "kernel": "HammingWeightClaimReductionFixed29",
            "log_t": log_t,
            "rows": row_count,
            "device": {
                "name": &execution.compile.device.name,
                "max_buffer_length": execution.compile.device.max_buffer_length,
                "max_threadgroup_memory_length": execution.compile.device.max_threadgroup_memory_length,
                "recommended_max_working_set_size": execution.compile.device.recommended_max_working_set_size,
                "offset": execution.compile.device.offset,
            },
            "pipeline_admitted": execution.compile.admitted(),
            "histogram_limits": {
                "thread_execution_width": execution.compile.histogram.thread_execution_width,
                "max_total_threads_per_threadgroup": execution.compile.histogram.max_total_threads_per_threadgroup,
                "static_threadgroup_memory_length": execution.compile.histogram.static_threadgroup_memory_length,
            },
            "finalize_limits": {
                "thread_execution_width": execution.compile.finalize.thread_execution_width,
                "max_total_threads_per_threadgroup": execution.compile.finalize.max_total_threads_per_threadgroup,
                "static_threadgroup_memory_length": execution.compile.finalize.static_threadgroup_memory_length,
            },
            "dynamic_threadgroup_memory_bytes": execution.compile.dynamic_threadgroup_memory_bytes,
            "library_compile_wall_ns": execution.compile.library_compile_wall.as_nanos() as u64,
            "row_build_wall_ns": row_build_wall.as_nanos() as u64,
            "execution_wall_ns": execution_wall.as_nanos() as u64,
            "gpu_active_ns": gpu_active_ns,
            "rows_per_second": (row_count as u128 * 1_000_000_000 / u128::from(gpu_active_ns)) as u64,
            "target_scale": target_scale,
            "target_gpu_active_ns": HAMMING_WEIGHT_TARGET_GPU_ACTIVE_NS,
            "five_x_cap_ns": HAMMING_WEIGHT_TARGET_FIVE_X_NS,
            "eight_x_cap_ns": HAMMING_WEIGHT_TARGET_EIGHT_X_NS,
            "clears_gpu_active_target": target_scale && gpu_active_ns <= HAMMING_WEIGHT_TARGET_GPU_ACTIVE_NS,
            "clears_five_x_gpu_active": target_scale && gpu_active_ns <= HAMMING_WEIGHT_TARGET_FIVE_X_NS,
            "clears_eight_x_gpu_active": target_scale && gpu_active_ns <= HAMMING_WEIGHT_TARGET_EIGHT_X_NS,
            "census": {
                "rows": execution.census.rows,
                "pc_present": execution.census.pc_present,
                "ram_present": execution.census.ram_present,
                "retained_nonzero_contributions": execution.census.retained_nonzero_contributions,
                "occupied_outer_bins": execution.census.occupied_outer_bins,
            },
        })
    );
    Ok(())
}

fn fixture_row(index: usize) -> HammingWeightResidentRow {
    let lookup_lo = (index as u64).wrapping_mul(0x0102_0304_0506_0708);
    let lookup_hi = (!(index as u64)).rotate_left(17);
    let ram = if index.is_multiple_of(3) {
        0
    } else {
        (index & 0xffff) as u64 + 1
    };
    let magnitude = (index as u64).wrapping_mul(0x1_0001);
    let pc = if index.is_multiple_of(5) {
        0
    } else {
        ((index * 7) & 0xffff) as u64 + 1
    };
    let negative = u64::from(index.is_multiple_of(7)) << 63;
    HammingWeightResidentRow::from_words([lookup_lo, lookup_hi, ram, magnitude, pc | negative])
}
