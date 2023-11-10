use std::cell::RefCell;

#[derive(Debug)]
pub struct TraceRow {
    pub instruction: Instruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
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
    pub rd_post_val: Option<i64>,
}

#[derive(Debug)]
pub enum MemoryState {
    Read {
        address: u64,
        value: u64,
    },
    Write {
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

pub struct Tracer {
    pub rows: RefCell<Vec<TraceRow>>,
    open: RefCell<bool>,
}

impl Tracer {
    pub fn new() -> Self {
        Self { rows: RefCell::new(Vec::new()), open: RefCell::new(false) }
    }

    pub fn start_instruction(&self, inst: Instruction) {
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
            row.register_state.rd_pre_val = Some(reg[rd]);
        }
    }

    pub fn capture_post_state(&self, reg: [i64; 32]) {
        if !*self.open.try_borrow().unwrap() {
            return;
        }

        let mut rows = self.rows.try_borrow_mut().unwrap();
        let row = rows.last_mut().unwrap();

        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_post_val = Some(reg[rd]);
        }

        if let Some(rs1) = row.instruction.rs1 {
            row.register_state.rs1_val = Some(reg[rs1]);
        }

        if let Some(rs2) = row.instruction.rs2 {
            row.register_state.rs2_val = Some(reg[rs2]);
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

