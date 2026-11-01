//! Count the complete external verifier without retaining a witness trace.
use anyhow::{anyhow, ensure, Result};
use jolt_host::Program;
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
    let mut program = Program::new("jolt-akita-external-guest");
    program.set_func("verify_external");
    program.set_std(true);
    program.set_heap_size(1 << 28);
    program.set_stack_size(1 << 25);
    program.set_max_input_size(4096);
    program.set_max_output_size(1 << 20);
    program.set_max_untrusted_advice_size(1 << 26);
    program.set_linker_memory_size(1 << 31);
    let (_, _, size, _) = program.decode();
    let memory = program.memory_config_with_program_size(size);
    let elf = program
        .elf
        .as_ref()
        .ok_or_else(|| anyhow!("missing guest ELF"))?;
    let advice = fs::read(dir.join("advice.bin"))?;
    let (cycles, io, _) = tracer::execute(
        elf.bytes(),
        Some(elf.path()),
        &[],
        &[],
        &advice,
        &memory,
        None,
    );
    ensure!(!io.panic, "external guest panicked");
    let expected = fs::read(dir.join("expected-output.bin"))?;
    ensure!(io.outputs == expected, "changed external output");
    let report = serde_json::json!({"cycles":cycles,"padded_trace":cycles.checked_next_power_of_two().ok_or_else(||anyhow!("trace size overflow"))?,"output_identical":true});
    fs::write(dir.join("cycles.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
