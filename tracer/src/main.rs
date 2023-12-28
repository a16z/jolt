extern crate tracer;

use std::env;

use common::path::JoltPaths;
use common::serializable::Serializable;
use tracer::{decode, trace};

pub fn run_tracer(program_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let elf_location = JoltPaths::elf_path(program_name);
    let trace_destination = JoltPaths::trace_path(program_name);
    let bytecode_destination = JoltPaths::bytecode_path(program_name);

    if !elf_location.exists() {
        return Err(format!("Could not find ELF file at location {:?}", elf_location).into());
    }

    let rows = trace(&elf_location);
    rows.serialize_to_file(&trace_destination)?;
    println!(
        "Wrote {} rows to         {}.",
        rows.len(),
        trace_destination.display()
    );

    let instructions = decode(&elf_location);
    instructions.serialize_to_file(&bytecode_destination)?;
    println!(
        "Wrote {} instructions to {}.",
        instructions.len(),
        bytecode_destination.display()
    );
    Ok(())
}

pub fn main() {
    // Note: assumes program is already compiled
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <program_name>", args[0]);
        std::process::exit(1);
    }
    let program_name = &args[1];

    if let Err(e) = run_tracer(program_name) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
