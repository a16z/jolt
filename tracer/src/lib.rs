#![allow(dead_code)]

use std::{path::PathBuf, fs::File, io::Read};

use emulator::{default_terminal::DefaultTerminal, Emulator, cpu};

mod trace;
mod emulator;

pub use common::{TraceRow, Instruction, RegisterState, MemoryState};

pub fn trace(elf: PathBuf) -> Vec<TraceRow> {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(cpu::Xlen::Bit32);

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

