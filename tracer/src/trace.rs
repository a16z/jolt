use std::{fs::File, io::Read, path::PathBuf};

use crate::emulator::{cpu, default_terminal::DefaultTerminal, elf_analyzer, Emulator};

pub fn trace(elf: PathBuf) {
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

    if prev_pc == pc {
      break;
    }

    prev_pc = pc;
  }

  let trace = &emulator.get_cpu().trace;
  for step in trace {
    println!("{:?}", step);
    println!();
  }
}

pub fn decode(elf: PathBuf) {
  let mut elf_file = File::open(elf).unwrap();

  let mut elf_contents = Vec::new();
  elf_file.read_to_end(&mut elf_contents).unwrap();

  let program = elf_analyzer::ElfAnalyzer::new(elf_contents);
}
