#![allow(dead_code)]

use std::{path::PathBuf, fs::File, io::Read};

use emulator::{default_terminal::DefaultTerminal, Emulator, cpu::{self, Xlen}};
use common;

use object::{Object, ObjectSection, SectionKind};

mod trace;
mod decode;
mod emulator;

pub use common::{RVTraceRow, Instruction, RegisterState, MemoryState};

use crate::decode::decode_raw;

pub fn trace(elf: PathBuf) -> Vec<RVTraceRow> {
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

pub fn decode(elf: PathBuf) -> Vec<Instruction> {
    let mut elf_file = File::open(elf).unwrap();
    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    let obj = object::File::parse(&*elf_contents).unwrap();

    let text_sections = obj
        .sections()
        .filter(|s| s.kind() == SectionKind::Text)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    for section in text_sections {
        let data = section.data().unwrap();

        for (chunk, word) in data.chunks(4).enumerate() {
            let word = u32::from_le_bytes(word.try_into().unwrap());
            let address = chunk as u64 * 4 + section.address();
            let inst = decode_raw(word).unwrap();
            let trace = inst.trace.unwrap();

            let inst = trace(&inst, &get_xlen(), word, address);
            instructions.push(inst);
        }
    }

    instructions
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers.")
    }
}