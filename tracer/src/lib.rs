#![allow(dead_code)]

use std::{fs::File, io::Read, path::PathBuf};

use common::{self, constants::RAM_START_ADDRESS, serializable::Serializable};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
};

use object::{Object, ObjectSection};

mod decode;
mod emulator;
mod trace;

pub use common::rv_trace::{
    ELFInstruction, JoltDevice, MemoryState, RVTraceRow, RegisterState, RV32IM,
};

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
    device_destination: PathBuf,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    if !elf_location.exists() {
        return Err(format!("Could not find ELF file at location {:?}", elf_location).into());
    }

    let (rows, device) = trace(&elf_location, Vec::new());
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

    device.serialize_to_file(&device_destination)?;
    println!(
        "Wrote {} bytes of inputs and outputs to {}.",
        device.size(),
        device_destination.display()
    );

    // let rows: Vec<()> = vec![];
    Ok((rows.len(), instructions.len()))
}

pub fn trace(elf: &PathBuf, inputs: Vec<u8>) -> (Vec<RVTraceRow>, JoltDevice) {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new();
    jolt_device.inputs = inputs;
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

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
    drop(rows);

    let device = emulator.get_mut_cpu().get_mut_mmu().jolt_device.clone();

    (output, device)
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
                opcode: RV32IM::from_str("UNIMPL"),
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
