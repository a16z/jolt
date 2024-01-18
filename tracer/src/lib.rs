#![allow(dead_code)]

use std::{fs::File, io::Read, path::PathBuf};

use common::{self, serializable::Serializable, Section};
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
/// * A `Result` containing a tuple of the num trace rows and sections if successful, or an error if not.
pub fn run_tracer_with_paths(
    elf_location: PathBuf,
    trace_destination: PathBuf,
    bytecode_destination: PathBuf,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
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

    let sections = decode(&elf_location);
    sections.serialize_to_file(&bytecode_destination)?;
    println!(
        "Wrote {} sections to {}.",
        sections.len(),
        bytecode_destination.display()
    );
    Ok((rows.len(), sections.len()))
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

pub fn decode(elf: &PathBuf) -> Vec<Section> {
    let mut elf_file = File::open(elf).unwrap();
    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    let obj = object::File::parse(&*elf_contents).unwrap();

    let mut data = obj
        .sections()
        .filter(|s| s.kind() == SectionKind::Data)
        .map(|s| Section::Data { data: s.data().unwrap().to_vec(), address: s.address(), size: s.size() })
        .collect::<Vec<_>>();

    let mut rodata = obj
        .sections()
        .filter(|s| s.kind() == SectionKind::ReadOnlyData)
        .map(|s| Section::RoData { data: s.data().unwrap().to_vec(), address: s.address(), size: s.size() })
        .collect::<Vec<_>>();

    let text_sections_raw = obj
        .sections()
        .filter(|s| s.kind() == SectionKind::Text)
        .collect::<Vec<_>>();

    let mut text = Vec::new();
    for section in text_sections_raw {
        let data = section.data().unwrap();

        let mut instructions = Vec::new();
        for (chunk, word) in data.chunks(4).enumerate() {
            let word = u32::from_le_bytes(word.try_into().unwrap());
            let address = chunk as u64 * 4 + section.address();
            let inst = decode_raw(word).unwrap();

            if let Some(trace) = inst.trace {
                let inst = trace(&inst, &get_xlen(), word, address);
                instructions.push(inst);
            } else {
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

        text.push(Section::Text { instructions, address: section.address(), size: section.size() });
    }

    let mut sections = Vec::new();
    sections.append(&mut text);
    sections.append(&mut data);
    sections.append(&mut rodata);

    sections
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
