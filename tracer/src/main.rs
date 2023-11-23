extern crate tracer;

use std::{env, path::PathBuf};

use tracer::{trace, decode};
use common::serializable::Serializable;

pub fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Note: ../target paths are hacks because we don't have access to the workspace root programatically
    let elf_location = root.join("../target/riscv32i-unknown-none-elf/release/fibonacci").canonicalize().unwrap();
    let trace_destination = root.join("../target/traces/trace.jolt").canonicalize().unwrap();
    let instruction_destination = root.join("../target/traces/elf.jolt").canonicalize().unwrap();

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
