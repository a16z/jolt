use std::{fs::File, io::Read, path::PathBuf};

use crate::emulator::{cpu, default_terminal::DefaultTerminal, Emulator};
 
#[derive(Debug)]
pub struct TraceRow {
    pub instruction: Instruction,
    pub register_state: RegisterState,
}

#[derive(Debug)]
pub struct Instruction {
    pub address: u64,
    pub opcode: &'static str,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub rd: Option<usize>,
    pub imm: Option<i128>,
}

#[derive(Debug, Clone, Default)]
pub struct RegisterState {
    pub rs1_val: Option<i64>,
    pub rs2_val: Option<i64>,
    pub rd_pre_val: Option<i64>,
    pub rd_val: Option<i64>,
}

pub struct Tracer {
    pub rows: Vec<TraceRow>,
}

impl Tracer {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn start_instruction(&mut self, inst: Instruction) {
        self.rows.push(TraceRow {
            instruction: inst,
            register_state: RegisterState::default(),
        });
    }

    pub fn capture_pre_state(&mut self, reg: [i64; 32]) {
        let row = self.rows.last_mut().unwrap();
        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_pre_val = Some(reg[rd]);
        }
    }

    pub fn capture_post_state(&mut self, reg: [i64; 32]) {
        let row = self.rows.last_mut().unwrap();

        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_val = Some(reg[rd]);
        }

        if let Some(rs1) = row.instruction.rs1 {
            row.register_state.rs1_val = Some(reg[rs1]);
        }

        if let Some(rs2) = row.instruction.rs2 {
            row.register_state.rs2_val = Some(reg[rs2]);
        }
    }
}

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

    let trace = &emulator.get_cpu().tracer.rows;
    for step in trace {
        println!("{:?}", step);
        println!();
    }
}

