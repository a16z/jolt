use common::constants::{RISCV_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize = 8;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

pub const CSR_MSTATUS: u16 = 0x300;
pub const CSR_MTVEC: u16 = 0x305;
pub const CSR_MSCRATCH: u16 = 0x340;
pub const CSR_MEPC: u16 = 0x341;
pub const CSR_MCAUSE: u16 = 0x342;
pub const CSR_MTVAL: u16 = 0x343;

const RESERVATION_W_REGISTER: u8 = RISCV_REGISTER_BASE;
const RESERVATION_D_REGISTER: u8 = RISCV_REGISTER_BASE + 1;

const TRAP_HANDLER_REGISTER: u8 = RISCV_REGISTER_BASE + 2;
const MSCRATCH_REGISTER: u8 = RISCV_REGISTER_BASE + 3;
const MEPC_REGISTER: u8 = RISCV_REGISTER_BASE + 4;
const MCAUSE_REGISTER: u8 = RISCV_REGISTER_BASE + 5;
const MTVAL_REGISTER: u8 = RISCV_REGISTER_BASE + 6;
const MSTATUS_REGISTER: u8 = RISCV_REGISTER_BASE + 7;

const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 8;

#[derive(Debug, Clone)]
pub struct VirtualRegisterAllocator {
    allocated: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
    pending_clearing_inline: Arc<Mutex<Vec<u8>>>,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
            pending_clearing_inline: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn reservation_w_register(&self) -> u8 {
        RESERVATION_W_REGISTER
    }

    pub fn reservation_d_register(&self) -> u8 {
        RESERVATION_D_REGISTER
    }

    pub fn trap_handler_register(&self) -> u8 {
        TRAP_HANDLER_REGISTER
    }

    pub fn mscratch_register(&self) -> u8 {
        MSCRATCH_REGISTER
    }

    pub fn mepc_register(&self) -> u8 {
        MEPC_REGISTER
    }

    pub fn mcause_register(&self) -> u8 {
        MCAUSE_REGISTER
    }

    pub fn mtval_register(&self) -> u8 {
        MTVAL_REGISTER
    }

    pub fn mstatus_register(&self) -> u8 {
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

    /// True when no virtual-register guards are outstanding: all transient
    /// registers are free and no inline clears are pending. Guards are
    /// strictly intra-tick, so this holds at every tick boundary — chunk
    /// checkpoints must only be captured in this state.
    pub fn is_quiescent(&self) -> bool {
        self.allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter()
            .all(|allocated| !*allocated)
            && self
                .pending_clearing_inline
                .lock()
                .expect("Failed to lock virtual register allocator")
                .is_empty()
    }

    #[cfg(test)]
    pub(crate) fn allocate(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(NUM_RESERVED_VIRTUAL_REGISTERS)
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        // Take 8 registers (40-47)
        {
            if !*allocated {
                *allocated = true;
                return VirtualRegisterGuard {
                    index: i as u8 + RISCV_REGISTER_BASE,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for instruction: No registers left");
    }

    /// Allocate virtual register that can be used in an inline.
    /// Uses registers 48+ (skips reserved 32-39 and instruction 40-47).
    ///
    /// A register may be allocated multiple times (e.g., separately by advice and inline
    /// sequence), but only cleared once.
    pub fn allocate_for_inline(&self) -> VirtualRegisterGuard {
        let skip_count = NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(skip_count)
        {
            if !*allocated {
                *allocated = true;
                let reg = i as u8 + RISCV_REGISTER_BASE;
                let mut pending = self
                    .pending_clearing_inline
                    .lock()
                    .expect("Failed to lock virtual register allocator");
                if !pending.contains(&reg) {
                    pending.push(reg);
                }
                return VirtualRegisterGuard {
                    index: reg,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for inline: No registers left");
    }

    pub fn get_registers_for_reset(&self) -> Vec<u8> {
        assert!(
            self.allocated
                .lock()
                .expect("Failed to lock virtual register allocator")
                .iter()
                .skip(NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS)
                .all(|allocated| !*allocated),
            "All inline virtual registers must be dropped before inline finalization"
        );

        std::mem::take(
            &mut self
                .pending_clearing_inline
                .lock()
                .expect("Failed to lock virtual register allocator"),
        )
    }

    fn deallocate(&self, index: u8) {
        let virtual_index = (index - RISCV_REGISTER_BASE) as usize;
        if virtual_index < NUM_VIRTUAL_REGISTERS {
            self.allocated
                .lock()
                .expect("Failed to lock virtual register allocator")[virtual_index] = false;
        }
    }
}

impl Default for VirtualRegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns whether `csr_addr` maps to a CSR that Jolt models in its proof
/// system. Used by `Instruction::decode` to reject unsupported CSR
/// instructions before they reach the tracer's inline-sequence path, which
/// would otherwise panic on unsupported CSRs.
pub fn is_supported_csr(csr_addr: u16) -> bool {
    matches!(
        csr_addr,
        CSR_MSTATUS | CSR_MTVEC | CSR_MSCRATCH | CSR_MEPC | CSR_MCAUSE | CSR_MTVAL
    )
}

pub struct VirtualRegisterGuard {
    index: u8,
    allocator: VirtualRegisterAllocator,
}

impl Deref for VirtualRegisterGuard {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Drop for VirtualRegisterGuard {
    fn drop(&mut self) {
        self.allocator.deallocate(self.index);
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
    fn test_allocate_deallocate() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate();
            assert_eq!(*guard1, FIRST_ALLOC_REG);

            let guard2 = allocator.allocate();
            assert_eq!(*guard2, FIRST_ALLOC_REG + 1);
        }

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG);
    }

    #[test]
    fn test_deref() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_ALLOC_REG);
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        for i in 0..NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            let guard = allocator.allocate();
            assert_eq!(*guard, FIRST_ALLOC_REG + i as u8);
            guards.push(guard);
        }

        let _guard = allocator.allocate();
    }

    #[test]
    fn test_allocate_deallocate_inline() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate_for_inline();
            assert_eq!(*guard1, FIRST_INLINE_REG);

            let guard2 = allocator.allocate_for_inline();
            assert_eq!(*guard2, FIRST_INLINE_REG + 1);
        }

        let guard3 = allocator.allocate_for_inline();
        assert_eq!(*guard3, FIRST_INLINE_REG);
    }

    #[test]
    fn test_deref_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate_for_inline();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_INLINE_REG);
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        let num_inline_registers = NUM_VIRTUAL_REGISTERS
            - NUM_RESERVED_VIRTUAL_REGISTERS
            - NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        for i in 0..num_inline_registers {
            let guard = allocator.allocate_for_inline();
            assert_eq!(*guard, FIRST_INLINE_REG + i as u8);
            guards.push(guard);
        }

        let _guard = allocator.allocate_for_inline();
    }

    #[test]
    fn test_combined_allocate_and_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let guard1 = allocator.allocate();
        assert_eq!(*guard1, FIRST_ALLOC_REG);

        let guard2 = allocator.allocate();
        assert_eq!(*guard2, FIRST_ALLOC_REG + 1);

        let inline_guard1 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard1, FIRST_INLINE_REG);

        let inline_guard2 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard2, FIRST_INLINE_REG + 1);

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG + 2);

        drop(guard2);
        drop(inline_guard1);

        let guard4 = allocator.allocate();
        assert_eq!(*guard4, FIRST_ALLOC_REG + 1);

        let inline_guard3 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard3, FIRST_INLINE_REG);
    }

    #[test]
    fn test_csr_to_virtual_register() {
        let allocator = VirtualRegisterAllocator::new();

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
