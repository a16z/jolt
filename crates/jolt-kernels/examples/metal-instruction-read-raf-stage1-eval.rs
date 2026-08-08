#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error};

use jolt_kernels::metal::solinas::{run_instruction_read_raf_stage1_probe, SolinasMetal};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let log_rows = argument(1, 12)?;
    let threads = argument(2, 256)?;
    let fused = argument(3, 0)? != 0;
    let context = SolinasMetal::for_akita()?;
    let result = run_instruction_read_raf_stage1_probe(&context, log_rows, threads, fused)?;
    println!(
        "{}",
        json!({
            "schema": "instruction_read_raf_stage1_probe_v1",
            "rows": result.rows,
            "threads_per_threadgroup": result.threads_per_threadgroup,
            "fused_grouped_phase": result.fused_grouped_phase,
            "scatter_gpu_active_ns": result.scatter_gpu_active.as_nanos(),
            "scatter_wall_ns": result.scatter_wall.as_nanos(),
            "baseline_phase_gpu_active_ns": result.baseline_phase_gpu_active.as_nanos(),
            "resident_phase_gpu_active_ns": result.resident_phase_gpu_active.as_nanos(),
            "exact": result.exact,
            "device": context.device_info().name,
        })
    );
    Ok(())
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(env::args()
        .nth(index)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}
