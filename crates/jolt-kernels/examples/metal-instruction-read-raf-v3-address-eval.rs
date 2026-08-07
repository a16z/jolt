#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error};

use jolt_kernels::metal::solinas::instruction_read_raf_v3::run_address_atom_probe;
use jolt_kernels::metal::solinas::SolinasMetal;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let log_rows = env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let context = SolinasMetal::for_akita()?;
    let result = run_address_atom_probe(&context, log_rows)?;
    println!(
        "{}",
        json!({
            "schema": "instruction_read_raf_v3_address_probe_v1",
            "rows": result.rows,
            "atoms": result.atoms,
            "phases": result.phases,
            "all_exact": result.all_exact,
            "finished": result.finished,
            "gpu_active_ns": result.gpu_active_ns,
            "device": context.device_info().name,
        })
    );
    Ok(())
}
