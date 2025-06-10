#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(clippy::legacy_numeric_constants)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::RAM_START_ADDRESS};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
};

use object::{Object, ObjectSection, SectionKind};

mod decode;
pub mod emulator;
mod trace;

use crate::decode::decode_raw;
use common::rv_trace::MemoryConfig;
pub use common::rv_trace::{
    ELFInstruction, JoltDevice, MemoryState, RVTraceRow, RegisterState, RV32IM,
};

#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
) -> (Vec<RVTraceRow>, JoltDevice) {
    let term = DefaultTerminal::default();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

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

#[tracing::instrument(skip_all)]
pub fn decode(elf: &[u8]) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
    let obj = object::File::parse(elf).unwrap();

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    let mut data = Vec::new();

    for section in sections {
        let raw_data = section.data().unwrap();

        if let SectionKind::Text = section.kind() {
            for (chunk, word) in raw_data.chunks(4).enumerate() {
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
                    opcode: RV32IM::UNIMPL,
                    rs1: None,
                    rs2: None,
                    rd: None,
                    imm: None,
                    virtual_sequence_remaining: None,
                });
            }
        }
        let address = section.address();
        for (offset, byte) in raw_data.iter().enumerate() {
            data.push((address + offset as u64, *byte));
        }
    }

    (instructions, data)
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
