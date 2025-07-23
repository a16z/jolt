use common::{constants::REGISTER_COUNT, rv_trace::*};
use std::cell::RefCell;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TracerError {
    #[error("Tracer is already borrowed")]
    BorrowError,
    #[error("No active instruction")]
    NoActiveInstruction,
}

type TracerResult<T> = Result<T, TracerError>;

#[derive(Default)]
pub struct Tracer {
    pub rows: RefCell<Vec<RVTraceRow>>,
    open: RefCell<bool>,
}

impl Tracer {
    pub fn start_instruction(&self, inst: ELFInstruction) -> TracerResult<()> {
        let mut inst = inst;
        inst.address = inst.address as u32 as u64;
        
        *self.open.try_borrow_mut().map_err(|_| TracerError::BorrowError)? = true;
        self.rows.try_borrow_mut().map_err(|_| TracerError::BorrowError)?.push(RVTraceRow {
            instruction: inst,
            register_state: RegisterState::default(),
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });
        Ok(())
    }

    pub fn capture_pre_state(&self, reg: [i64; 32], xlen: &Xlen) -> TracerResult<()> {
        if !*self.open.try_borrow().map_err(|_| TracerError::BorrowError)? {
            return Ok(());
        }

        let mut rows = self.rows.try_borrow_mut().map_err(|_| TracerError::BorrowError)?;
        let row = rows.last_mut().ok_or(TracerError::NoActiveInstruction)?;

        if let Some(rs1) = row.instruction.rs1 {
            row.register_state.rs1_val = Some(normalize_register_value(reg[rs1 as usize], xlen));
        }

        if let Some(rs2) = row.instruction.rs2 {
            row.register_state.rs2_val = Some(normalize_register_value(reg[rs2 as usize], xlen));
        }
        Ok(())
    }

    pub fn capture_post_state(&self, reg: [i64; 32], xlen: &Xlen) -> TracerResult<()> {
        if !*self.open.try_borrow().map_err(|_| TracerError::BorrowError)? {
            return Ok(());
        }

        let mut rows = self.rows.try_borrow_mut().map_err(|_| TracerError::BorrowError)?;
        let row = rows.last_mut().ok_or(TracerError::NoActiveInstruction)?;

        if let Some(rd) = row.instruction.rd {
            row.register_state.rd_post_val = Some(normalize_register_value(reg[rd as usize], xlen));
        }
        Ok(())
    }

    pub fn push_memory(&self, memory_state: MemoryState) -> TracerResult<()> {
        if !*self.open.try_borrow().map_err(|_| TracerError::BorrowError)? {
            return Ok(());
        }

        if let Some(row) = self.rows.try_borrow_mut().map_err(|_| TracerError::BorrowError)?.last_mut() {
            row.memory_state = Some(memory_state);
        }
        Ok(())
    }

    pub fn end_instruction(&self) -> TracerResult<()> {
        *self.open.try_borrow_mut().map_err(|_| TracerError::BorrowError)? = false;
        Ok(())
    }
}

fn normalize_register_value(value: i64, xlen: &Xlen) -> u64 {
    match xlen {
        Xlen::Bit32 => value as u32 as u64,
        Xlen::Bit64 => value as u64,
    }
}
