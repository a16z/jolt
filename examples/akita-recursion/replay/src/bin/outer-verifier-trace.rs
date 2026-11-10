//! Count the complete external verifier without retaining a witness trace.
#[path = "../../../guest_build.rs"]
mod guest_build;
use anyhow::{anyhow, ensure, Result};
use common::constants::RAM_START_ADDRESS;
use common::jolt_device::MemoryConfig;
use std::{fs, path::PathBuf};
use tracing::Level;
fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 2,
        "usage: outer-verifier-trace PREPARED_DIRECTORY"
    );
    let dir = PathBuf::from(&args[1]);
    std::env::set_var("JOLT_AKITA_EXTERNAL_MATERIAL", dir.canonicalize()?);
    let mut memory = MemoryConfig {
        heap_size: 1 << 28,
        stack_size: 1 << 25,
        max_input_size: 4096,
        max_output_size: 1 << 20,
        max_untrusted_advice_size: 1 << 26,
        ..Default::default()
    };
    let elf = guest_build::build(
        "jolt-akita-external-guest",
        "verify_external",
        &memory,
        1 << 31,
    )?;
    let (_, _, program_end, _) = tracer::decode(elf.bytes());
    memory.program_size = Some(program_end - RAM_START_ADDRESS);
    let advice = fs::read(dir.join("advice.bin"))?;
    let (cycles, io, _) = tracer::execute(
        elf.bytes(),
        Some(elf.path()),
        &[],
        &advice,
        &[],
        &memory,
        None,
    );
    ensure!(!io.panic, "external guest panicked");
    let expected = fs::read(dir.join("expected-output.bin"))?;
    let (output, padding) = io
        .outputs
        .split_at_checked(expected.len())
        .ok_or_else(|| anyhow!("truncated external output"))?;
    ensure!(
        output == expected && padding.iter().all(|byte| *byte == 0),
        "changed external output"
    );
    let report = serde_json::json!({"cycles":cycles,"padded_trace":cycles.checked_next_power_of_two().ok_or_else(||anyhow!("trace size overflow"))?,"output_identical":true});
    fs::write(dir.join("cycles.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
