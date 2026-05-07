use common::constants::{RISCV_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT};

use crate::expand::ExpansionError;

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize = 8;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;
const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 8;

const RESERVATION_W_REGISTER: u8 = RISCV_REGISTER_BASE;
const RESERVATION_D_REGISTER: u8 = RISCV_REGISTER_BASE + 1;
const TRAP_HANDLER_REGISTER: u8 = RISCV_REGISTER_BASE + 2;
const MSCRATCH_REGISTER: u8 = RISCV_REGISTER_BASE + 3;
const MEPC_REGISTER: u8 = RISCV_REGISTER_BASE + 4;
const MCAUSE_REGISTER: u8 = RISCV_REGISTER_BASE + 5;
const MTVAL_REGISTER: u8 = RISCV_REGISTER_BASE + 6;
const MSTATUS_REGISTER: u8 = RISCV_REGISTER_BASE + 7;

pub const CSR_MSTATUS: u16 = 0x300;
pub const CSR_MTVEC: u16 = 0x305;
pub const CSR_MSCRATCH: u16 = 0x340;
pub const CSR_MEPC: u16 = 0x341;
pub const CSR_MCAUSE: u16 = 0x342;
pub const CSR_MTVAL: u16 = 0x343;

#[derive(Debug, Clone)]
pub struct ExpansionAllocator {
    allocated: [bool; NUM_VIRTUAL_REGISTERS],
    /// Inline-only virtual registers that must be reset before finalizing an inline sequence.
    pending_clearing_inline: Vec<u8>,
}

impl ExpansionAllocator {
    pub const fn new() -> Self {
        Self {
            allocated: [false; NUM_VIRTUAL_REGISTERS],
            pending_clearing_inline: Vec::new(),
        }
    }

    pub const fn reservation_w_register(&self) -> u8 {
        RESERVATION_W_REGISTER
    }

    pub const fn reservation_d_register(&self) -> u8 {
        RESERVATION_D_REGISTER
    }

    pub const fn trap_handler_register(&self) -> u8 {
        TRAP_HANDLER_REGISTER
    }

    pub const fn mscratch_register(&self) -> u8 {
        MSCRATCH_REGISTER
    }

    pub const fn mepc_register(&self) -> u8 {
        MEPC_REGISTER
    }

    pub const fn mcause_register(&self) -> u8 {
        MCAUSE_REGISTER
    }

    pub const fn mtval_register(&self) -> u8 {
        MTVAL_REGISTER
    }

    pub const fn mstatus_register(&self) -> u8 {
        MSTATUS_REGISTER
    }

    pub fn csr_to_virtual_register(&self, csr_addr: u16) -> Option<u8> {
        match csr_addr {
            CSR_MSTATUS => Some(self.mstatus_register()),
            CSR_MTVEC => Some(self.trap_handler_register()),
            CSR_MSCRATCH => Some(self.mscratch_register()),
            CSR_MEPC => Some(self.mepc_register()),
            CSR_MCAUSE => Some(self.mcause_register()),
            CSR_MTVAL => Some(self.mtval_register()),
            _ => None,
        }
    }

    pub fn allocate(&mut self) -> Result<u8, ExpansionError> {
        self.allocate_in_range(
            NUM_RESERVED_VIRTUAL_REGISTERS,
            NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS,
            "instruction",
        )
    }

    pub fn allocate_for_inline(&mut self) -> Result<u8, ExpansionError> {
        let register = self.allocate_in_range(
            NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS,
            NUM_VIRTUAL_REGISTERS,
            "inline",
        )?;
        if !self.pending_clearing_inline.contains(&register) {
            self.pending_clearing_inline.push(register);
        }
        Ok(register)
    }

    pub fn release(&mut self, register: u8) -> Result<(), ExpansionError> {
        let index = Self::virtual_index(register)?;
        if !self.allocated[index] {
            return Err(ExpansionError::UnallocatedVirtualRegister { register });
        }
        self.allocated[index] = false;
        Ok(())
    }

    pub fn take_registers_for_reset(&mut self) -> Result<Vec<u8>, ExpansionError> {
        if self
            .allocated
            .iter()
            .skip(NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS)
            .any(|allocated| *allocated)
        {
            return Err(ExpansionError::InlineRegistersStillAllocated);
        }
        Ok(std::mem::take(&mut self.pending_clearing_inline))
    }

    fn allocate_in_range(
        &mut self,
        start: usize,
        end: usize,
        pool: &'static str,
    ) -> Result<u8, ExpansionError> {
        for index in start..end {
            if !self.allocated[index] {
                self.allocated[index] = true;
                return Ok(RISCV_REGISTER_BASE + index as u8);
            }
        }
        Err(ExpansionError::VirtualRegisterExhausted { pool })
    }

    fn virtual_index(register: u8) -> Result<usize, ExpansionError> {
        if register < RISCV_REGISTER_BASE {
            return Err(ExpansionError::InvalidVirtualRegister { register });
        }
        let index = (register - RISCV_REGISTER_BASE) as usize;
        if index >= NUM_VIRTUAL_REGISTERS {
            return Err(ExpansionError::InvalidVirtualRegister { register });
        }
        Ok(index)
    }
}

impl Default for ExpansionAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_ALLOC_REG: u8 = RISCV_REGISTER_BASE + NUM_RESERVED_VIRTUAL_REGISTERS as u8;
    const FIRST_INLINE_REG: u8 = RISCV_REGISTER_BASE
        + NUM_RESERVED_VIRTUAL_REGISTERS as u8
        + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8;

    #[test]
    fn allocates_and_reuses_instruction_registers() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let first = allocator.allocate()?;
        let second = allocator.allocate()?;
        assert_eq!(first, FIRST_ALLOC_REG);
        assert_eq!(second, FIRST_ALLOC_REG + 1);

        allocator.release(first)?;
        assert_eq!(allocator.allocate()?, FIRST_ALLOC_REG);
        Ok(())
    }

    #[test]
    fn allocates_inline_registers_and_returns_reset_list() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let first = allocator.allocate_for_inline()?;
        let second = allocator.allocate_for_inline()?;
        assert_eq!(first, FIRST_INLINE_REG);
        assert_eq!(second, FIRST_INLINE_REG + 1);

        allocator.release(first)?;
        allocator.release(second)?;
        assert_eq!(
            allocator.take_registers_for_reset()?,
            vec![FIRST_INLINE_REG, FIRST_INLINE_REG + 1]
        );
        assert!(allocator.take_registers_for_reset()?.is_empty());
        Ok(())
    }

    #[test]
    fn errors_if_inline_registers_are_still_live_at_reset() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let _register = allocator.allocate_for_inline()?;
        assert!(matches!(
            allocator.take_registers_for_reset(),
            Err(ExpansionError::InlineRegistersStillAllocated)
        ));
        Ok(())
    }

    #[test]
    fn reports_exhaustion_without_panicking() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        for _ in 0..NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            let _register = allocator.allocate()?;
        }
        assert!(matches!(
            allocator.allocate(),
            Err(ExpansionError::VirtualRegisterExhausted {
                pool: "instruction"
            })
        ));
        Ok(())
    }

    #[test]
    fn maps_supported_csrs_to_reserved_registers() {
        let allocator = ExpansionAllocator::new();
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MSTATUS),
            Some(MSTATUS_REGISTER)
        );
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MTVEC),
            Some(TRAP_HANDLER_REGISTER)
        );
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MSCRATCH),
            Some(MSCRATCH_REGISTER)
        );
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MEPC),
            Some(MEPC_REGISTER)
        );
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MCAUSE),
            Some(MCAUSE_REGISTER)
        );
        assert_eq!(
            allocator.csr_to_virtual_register(CSR_MTVAL),
            Some(MTVAL_REGISTER)
        );
        assert_eq!(allocator.csr_to_virtual_register(0x999), None);
    }
}
