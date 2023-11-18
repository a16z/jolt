use std::cell::RefCell;

use common::{TraceRow, Instruction, RegisterState, MemoryState};

pub struct Tracer {
    pub rows: RefCell<Vec<TraceRow>>,
    open: RefCell<bool>,
}

impl Tracer {
    pub fn new() -> Self {
        Self { rows: RefCell::new(Vec::new()), open: RefCell::new(false) }
    }

    pub fn start_instruction(&self, inst: Instruction) {
        let mut inst = inst;
        inst.address = inst.address as u32 as u64;
        *self.open.try_borrow_mut().unwrap() = true;
        self.rows.try_borrow_mut().unwrap().push(TraceRow {
            instruction: inst,
            register_state: RegisterState::default(),
            memory_state: None,
        });
    }

    pub fn capture_pre_state(&self, reg: [i64; 32]) {
        if !*self.open.try_borrow().unwrap() {
            return;
        }

        let mut rows = self.rows.try_borrow_mut().unwrap();
        let row = rows.last_mut().unwrap();
        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_pre_val = Some(reg[rd as usize] as u64);
        }
    }

    pub fn capture_post_state(&self, reg: [i64; 32]) {
        if !*self.open.try_borrow().unwrap() {
            return;
        }

        let mut rows = self.rows.try_borrow_mut().unwrap();
        let row = rows.last_mut().unwrap();

        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_post_val = Some(reg[rd as usize] as u64);
        }

        if let Some(rs1) = row.instruction.rs1 {
            row.register_state.rs1_val = Some(reg[rs1 as usize] as u64);
        }

        if let Some(rs2) = row.instruction.rs2 {
            row.register_state.rs2_val = Some(reg[rs2 as usize] as u64);
        }
    }

    pub fn push_memory(&self, memory_state: MemoryState) {
        if !*self.open.try_borrow().unwrap() {
            return;
        }

        if let Some(row) = self.rows.try_borrow_mut().unwrap().last_mut() {
            row.memory_state = Some(memory_state);
        }
    }

    pub fn end_instruction(&self) {
        *self.open.try_borrow_mut().unwrap() = false;
    }
}

