extern crate tracer;

use std::env;
use common::path::JoltPaths;
use tracer::run_tracer_with_paths;

pub fn run_tracer(program_name: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let elf_location = JoltPaths::elf_path(program_name);
    let trace_destination = JoltPaths::trace_path(program_name);
    let bytecode_destination = JoltPaths::bytecode_path(program_name);
    let device_destination = JoltPaths::jolt_device_path(program_name);

    run_tracer_with_paths(
        elf_location, 
        trace_destination,
        bytecode_destination, 
        device_destination
    )
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
