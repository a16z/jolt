extern crate tracer;

use std::env;

use tracer::{trace, decode};
use common::serializable::Serializable;
use common::path::JoltPaths;

pub fn main() {
    // Note: assumes program is already compiled
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <program_name>", args[0]);
        std::process::exit(1);
    }
    let program_name = &args[1];

    let elf_location = JoltPaths::elf_path(program_name);
    let trace_destination = JoltPaths::trace_path(program_name);
    let bytecode_destination = JoltPaths::bytecode_path(program_name);

    if !elf_location.exists() {
        println!("Could not find ELF file at location {:?}", elf_location);
    }

    let rows = trace(&elf_location);
    rows.serialize_to_file(&trace_destination).expect(format!("Failed to write to {}", trace_destination.display()).as_str());
    println!("Wrote {} rows to         {}.", rows.len(), trace_destination.display());

    let instructions = decode(&elf_location);
    instructions.serialize_to_file(&bytecode_destination).expect(format!("Failed to write to {}", bytecode_destination.display()).as_str());

    println!("Wrote {} instructions to {}.", instructions.len(), bytecode_destination.display());
}
