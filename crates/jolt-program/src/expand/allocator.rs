use jolt_common::constants::{RISCV_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT};

use crate::expand::ExpansionError;

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
pub(super) const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize = 8;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;
const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 8;
const FIRST_INLINE_REGISTER_INDEX: usize =
    NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS;
pub(super) const FIRST_INLINE_REGISTER: u8 =
    RISCV_REGISTER_BASE + FIRST_INLINE_REGISTER_INDEX as u8;
pub(super) const NUM_INLINE_REGISTERS: usize = NUM_VIRTUAL_REGISTERS - FIRST_INLINE_REGISTER_INDEX;
const MAX_RECURSION_DEPTH: usize = 128;

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

pub(super) const fn reservation_w_register() -> u8 {
    RESERVATION_W_REGISTER
}

pub(super) const fn reservation_d_register() -> u8 {
    RESERVATION_D_REGISTER
}

pub(super) const fn trap_handler_register() -> u8 {
    TRAP_HANDLER_REGISTER
}

pub(super) const fn mepc_register() -> u8 {
    MEPC_REGISTER
}

pub(super) const fn mcause_register() -> u8 {
    MCAUSE_REGISTER
}

pub(super) const fn mtval_register() -> u8 {
    MTVAL_REGISTER
}

pub(super) const fn mstatus_register() -> u8 {
    MSTATUS_REGISTER
}

pub(super) fn virtual_register_for_csr(csr_addr: u16) -> Option<u8> {
    match csr_addr {
        CSR_MSTATUS => Some(mstatus_register()),
        CSR_MTVEC => Some(trap_handler_register()),
        CSR_MSCRATCH => Some(mscratch_register()),
        CSR_MEPC => Some(mepc_register()),
        CSR_MCAUSE => Some(mcause_register()),
        CSR_MTVAL => Some(mtval_register()),
        _ => None,
    }
}

pub(super) const fn mscratch_register() -> u8 {
    MSCRATCH_REGISTER
}

/// Owns virtual-register assignment during bytecode expansion.
///
/// The pool is partitioned into reserved registers for modeled machine state,
/// short-lived instruction temps for built-in source-only recipes, and inline
/// registers for provider-built registered inline recipes. Keeping this state in
/// `jolt-program` is what makes static bytecode expansion and runtime tracing
/// agree on virtual-register numbers.
#[derive(Debug, Clone)]
pub struct ExpansionAllocator {
    allocated: u128,
    pending_clearing_inline: u128,
    recursion_depth: usize,
}

impl ExpansionAllocator {
    /// Create an empty allocator with no live virtual registers and no pending
    /// inline reset rows.
    pub const fn new() -> Self {
        Self {
            allocated: 0,
            pending_clearing_inline: 0,
            recursion_depth: 0,
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

    /// Allocate a short-lived instruction temporary from registers 40..=47.
    ///
    /// These registers are used for built-in source-only expansions and for
    /// top-level `rd = x0` rewrites. Callers must release the returned register
    /// before the current recipe finishes.
    pub fn allocate(&mut self) -> Result<u8, ExpansionError> {
        self.allocate_in_range(
            NUM_RESERVED_VIRTUAL_REGISTERS,
            NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS,
            "instruction",
        )
    }

    /// Allocate a registered-inline virtual register from the long-lived inline
    /// range.
    ///
    /// Inline registers are explicit builder resources: the recipe must release
    /// them, but the allocator still records that they were touched so
    /// `materialize_inline` can append reset rows at the end of the stamped
    /// inline sequence.
    pub fn allocate_for_inline(&mut self) -> Result<u8, ExpansionError> {
        let register =
            self.allocate_in_range(FIRST_INLINE_REGISTER_INDEX, NUM_VIRTUAL_REGISTERS, "inline")?;
        self.pending_clearing_inline |= Self::register_bit(register)?;
        Ok(register)
    }

    /// Release a live virtual register previously returned by this allocator.
    ///
    /// Releasing an unknown register is an error; this catches leaked or
    /// double-released recipe resources before final bytecode is accepted.
    pub fn release(&mut self, register: u8) -> Result<(), ExpansionError> {
        let bit = Self::register_bit(register)?;
        if self.allocated & bit == 0 {
            return Err(ExpansionError::UnallocatedVirtualRegister { register });
        }
        self.allocated &= !bit;
        Ok(())
    }

    /// Return and clear the inline registers that need reset rows.
    ///
    /// All inline registers must already be released. The returned list is
    /// sorted by register number so reset-row materialization is deterministic.
    pub fn take_registers_for_reset(&mut self) -> Result<Vec<u8>, ExpansionError> {
        let inline_mask = Self::range_mask(FIRST_INLINE_REGISTER_INDEX, NUM_VIRTUAL_REGISTERS);
        if self.allocated & inline_mask != 0 {
            return Err(ExpansionError::InlineRegistersStillAllocated);
        }
        let pending = self.pending_clearing_inline;
        self.pending_clearing_inline = 0;
        Ok(Self::registers_in_mask(pending))
    }

    pub(super) fn enter_expansion(&mut self) -> Result<(), ExpansionError> {
        if self.recursion_depth == MAX_RECURSION_DEPTH {
            return Err(ExpansionError::RecursionDepthExceeded {
                max_depth: MAX_RECURSION_DEPTH,
            });
        }
        self.recursion_depth += 1;
        Ok(())
    }

    pub(super) fn exit_expansion(&mut self) {
        self.recursion_depth -= 1;
    }

    fn allocate_in_range(
        &mut self,
        start: usize,
        end: usize,
        pool: &'static str,
    ) -> Result<u8, ExpansionError> {
        for index in start..end {
            let bit = 1u128 << index;
            if self.allocated & bit == 0 {
                self.allocated |= bit;
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

    fn register_bit(register: u8) -> Result<u128, ExpansionError> {
        Ok(1u128 << Self::virtual_index(register)?)
    }

    fn range_mask(start: usize, end: usize) -> u128 {
        let len = end - start;
        ((1u128 << len) - 1) << start
    }

    fn registers_in_mask(mask: u128) -> Vec<u8> {
        let mut registers = Vec::new();
        for index in 0..NUM_VIRTUAL_REGISTERS {
            if mask & (1u128 << index) != 0 {
                registers.push(RISCV_REGISTER_BASE + index as u8);
            }
        }
        registers
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
        assert_eq!(
            virtual_register_for_csr(CSR_MSTATUS),
            Some(MSTATUS_REGISTER)
        );
        assert_eq!(
            virtual_register_for_csr(CSR_MTVEC),
            Some(TRAP_HANDLER_REGISTER)
        );
        assert_eq!(
            virtual_register_for_csr(CSR_MSCRATCH),
            Some(MSCRATCH_REGISTER)
        );
        assert_eq!(virtual_register_for_csr(CSR_MEPC), Some(MEPC_REGISTER));
        assert_eq!(virtual_register_for_csr(CSR_MCAUSE), Some(MCAUSE_REGISTER));
        assert_eq!(virtual_register_for_csr(CSR_MTVAL), Some(MTVAL_REGISTER));
        assert_eq!(virtual_register_for_csr(0x999), None);
    }
}
