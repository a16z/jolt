#![allow(dead_code)]

use std::{fs::File, io::Read, path::PathBuf, collections::HashMap};

use common::{self, constants::RAM_START_ADDRESS, serializable::Serializable, RV32IM};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
};

use object::{Object, ObjectSection, SectionKind};

mod decode;
mod emulator;
mod trace;

pub use common::{ELFInstruction, MemoryState, RVTraceRow, RegisterState};

use crate::decode::decode_raw;

/// Runs the tracer with the provided paths.
///
/// # Parameters
///
/// * `elf_location`: The path to the ELF file.
/// * `trace_destination`: The path where the trace will be written.
/// * `bytecode_destination`: The path where the bytecode will be written.
///
/// # Returns
///
/// * A `Result` containing a tuple of the num trace rows and instructions if successful, or an error if not.
pub fn run_tracer_with_paths(
    elf_location: PathBuf,
    trace_destination: PathBuf,
    bytecode_destination: PathBuf,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    if !elf_location.exists() {
        return Err(format!("Could not find ELF file at location {:?}", elf_location).into());
    }

    let rows = trace(&elf_location);


    let mut instruction_count: HashMap<RV32IM, usize> = HashMap::new();
    for row in &rows {
        *instruction_count.entry(row.instruction.opcode).or_insert(0) += 1;
    }

    // Sort the instruction count by value in descending order and print them
    let mut instruction_count_vec: Vec<_> = instruction_count.iter().collect();
    instruction_count_vec.sort_by(|a, b| b.1.cmp(a.1));
    println!("Trace instruction counts:");
    for (opcode, count) in instruction_count_vec {
        println!("- {:?}: {}", opcode, count);
    }

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
    Ok((rows.len(), instructions.len()))
}

pub fn trace(elf: &PathBuf) -> Vec<RVTraceRow> {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut elf_file = File::open(elf).unwrap();

    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    emulator.setup_program(elf_contents);

    let mut prev_pc = 0;
    loop {
        let pc = emulator.get_cpu().read_pc();
        emulator.tick();

        // This is a trick to see if the program has terminated by throwing itself
        // into an infinite loop. It seems to be a good heuristic for now but we
        // should eventually migrate to an explicit shutdown signal.
        if prev_pc == pc {
            break;
        }

        prev_pc = pc;
    }

    let mut rows = emulator.get_mut_cpu().tracer.rows.try_borrow_mut().unwrap();
    let mut output = Vec::new();
    output.append(&mut rows);

    output
}

pub fn decode(elf: &PathBuf) -> Vec<ELFInstruction> {
    let mut elf_file = File::open(elf).unwrap();
    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    let obj = object::File::parse(&*elf_contents).unwrap();

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    for section in sections {
        let data = section.data().unwrap();

        for (chunk, word) in data.chunks(4).enumerate() {
            let word = u32::from_le_bytes(word.try_into().unwrap());
            let address = chunk as u64 * 4 + section.address();

            if let Ok(inst) = decode_raw(word) {
                if let Some(trace) = inst.trace {
                    let inst = trace(&inst, &get_xlen(), word, address);
                    instructions.push(inst);
                    continue;
                }
            }
            // Unrecognized instruction, or from a ReadOnlyData section
            instructions.push(ELFInstruction {
                address,
                opcode: common::RV32IM::from_str("UNIMPL"),
                raw: word,
                rs1: None,
                rs2: None,
                rd: None,
                imm: None,
            });
        }
    }

    instructions
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
