extern crate tracer;

use std::{env, path::PathBuf};

use tracer::{trace, decode};
use common::serializable::Serializable;
use common::path::JoltPaths;

pub fn main() {
    // Note: assumes program is already compiled
    let program_name = "fibonacci";
    let elf_location = JoltPaths::elf_path(program_name);
    let trace_destination = JoltPaths::trace_path(program_name);
    let instruction_destination = JoltPaths::elf_trace_path(program_name);

    if !elf_location.exists() {
        println!("Could not find ELF file at location {:?}", elf_location);
    }

    let rows = trace(&elf_location);
    rows.serialize_to_file(&trace_destination).expect(format!("Failed to write to {}", trace_destination.display()).as_str());
    println!("Wrote {} rows to         {}.", rows.len(), trace_destination.display());

    let instructions = decode(&elf_location);
    instructions.serialize_to_file(&instruction_destination).expect(format!("Failed to write to {}", instruction_destination.display()).as_str());

    println!("Wrote {} instructions to {}.", instructions.len(), instruction_destination.display());
}
