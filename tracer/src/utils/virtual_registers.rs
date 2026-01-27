use common::constants::{
    RISCV_REGISTER_COUNT, VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT,
};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize =
    VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT as usize;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

/// CSR addresses for M-mode CSRs supported by the virtual register system.
/// These are the standard RISC-V CSR addresses.
pub const CSR_MSTATUS: u16 = 0x300; // Machine Status
pub const CSR_MTVEC: u16 = 0x305; // Machine Trap-Vector Base Address
pub const CSR_MSCRATCH: u16 = 0x340; // Machine Scratch Register
pub const CSR_MEPC: u16 = 0x341; // Machine Exception Program Counter
pub const CSR_MCAUSE: u16 = 0x342; // Machine Trap Cause
pub const CSR_MTVAL: u16 = 0x343; // Machine Trap Value

/// Layout of virtual registers:
/// - Registers 32-38: Reserved CSR registers (persistent, never allocated)
///   - 32: Reservation address for LR/SC
///   - 33: mtvec (trap handler address)
///   - 34: mscratch (trap scratch register)
///   - 35: mepc (exception PC)
///   - 36: mcause (trap cause)
///   - 37: mtval (trap value)
///   - 38: mstatus (machine status)
/// - Registers 39-45: Temporary registers for inline sequences (allocate())
/// - Registers 46+: Registers for larger inlines (allocate_for_inline())
///
/// The reserved registers (32-38) are skipped by allocate() and allocate_for_inline()
/// to ensure they persist across instructions.
/// Register 32 is used for LR/SC reservation address.
const RESERVATION_REGISTER: u8 = RISCV_REGISTER_BASE; // register 32

/// CSR Virtual Register Mapping (persistent, not allocated):
/// - Register 33: mtvec (trap handler address)
/// - Register 34: mscratch (trap scratch register)
/// - Register 35: mepc (exception PC)
/// - Register 36: mcause (trap cause)
/// - Register 37: mtval (trap value)
/// - Register 38: mstatus (machine status)
const TRAP_HANDLER_REGISTER: u8 = RISCV_REGISTER_BASE + 1; // register 33 (mtvec)
const MSCRATCH_REGISTER: u8 = RISCV_REGISTER_BASE + 2; // register 34
const MEPC_REGISTER: u8 = RISCV_REGISTER_BASE + 3; // register 35
const MCAUSE_REGISTER: u8 = RISCV_REGISTER_BASE + 4; // register 36
const MTVAL_REGISTER: u8 = RISCV_REGISTER_BASE + 5; // register 37
const MSTATUS_REGISTER: u8 = RISCV_REGISTER_BASE + 6; // register 38

/// Number of reserved virtual registers that are NOT allocated.
/// Includes: reservation (32), mtvec (33), mscratch (34), mepc (35),
///           mcause (36), mtval (37), mstatus (38)
/// allocate() skips these and starts from register 39.
const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 7;

#[derive(Debug, Clone)]
pub struct VirtualRegisterAllocator {
    allocated: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
    /// At the end of the inline execution all registers have to be reset to 0
    /// This variable tracks which registers were allocated during inline execution
    pending_clearing_inline: Arc<Mutex<Vec<u8>>>,
    /// Tracks whether the current ECALL is a CSR ECALL (for trap handler setup).
    /// Set by cpu.handle_syscall(), read by ECALL.trace() to determine inline sequence.
    is_csr_ecall: Arc<Mutex<bool>>,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
            pending_clearing_inline: Arc::new(Mutex::new(Vec::new())),
            is_csr_ecall: Arc::new(Mutex::new(false)),
        }
    }

    /// Get the reservation register (register 32) used for LR/SC operations.
    /// This register holds the memory address reserved by LR instructions
    /// and is checked by SC instructions.
    pub fn reservation_register(&self) -> u8 {
        RESERVATION_REGISTER
    }

    /// Get the trap handler register (register 33) - mtvec CSR.
    /// Set at boot via CSR ECALL, used by trap-taking ECALLs to verify target.
    pub fn trap_handler_register(&self) -> u8 {
        TRAP_HANDLER_REGISTER
    }

    /// Get the mscratch register (register 34) - scratch register for trap handler.
    pub fn mscratch_register(&self) -> u8 {
        MSCRATCH_REGISTER
    }

    /// Get the mepc register (register 35) - exception program counter.
    pub fn mepc_register(&self) -> u8 {
        MEPC_REGISTER
    }

    /// Get the mcause register (register 36) - trap cause.
    pub fn mcause_register(&self) -> u8 {
        MCAUSE_REGISTER
    }

    /// Get the mtval register (register 37) - trap value.
    pub fn mtval_register(&self) -> u8 {
        MTVAL_REGISTER
    }

    /// Get the mstatus register (register 38) - machine status.
    pub fn mstatus_register(&self) -> u8 {
        MSTATUS_REGISTER
    }

    /// Map a CSR address to its corresponding virtual register number.
    /// Returns Some(register) if the CSR is supported, None otherwise.
    ///
    /// Supported CSRs:
    /// - mstatus (0x300) → vr38
    /// - mtvec (0x305) → vr33
    /// - mscratch (0x340) → vr34
    /// - mepc (0x341) → vr35
    /// - mcause (0x342) → vr36
    /// - mtval (0x343) → vr37
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

    /// Set whether the current ECALL is a CSR ECALL (for trap handler setup).
    /// Called by cpu.handle_syscall() to communicate with ECALL.trace().
    pub fn set_is_csr_ecall(&self, value: bool) {
        *self
            .is_csr_ecall
            .lock()
            .expect("Failed to lock is_csr_ecall") = value;
    }

    /// Get whether the current ECALL is a CSR ECALL.
    /// Called by ECALL.trace() to determine which inline sequence to use.
    pub fn is_csr_ecall(&self) -> bool {
        *self
            .is_csr_ecall
            .lock()
            .expect("Failed to lock is_csr_ecall")
    }

    /// Allocate virtual register that can be used in the inline sequence of
    /// an instruction. Skips reserved registers (32-38) and uses registers 39-45.
    pub(crate) fn allocate(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(NUM_RESERVED_VIRTUAL_REGISTERS) // Skip reserved registers (32-38)
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        // Take 7 registers (39-45)
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
    /// Uses registers 46+ (skips reserved 32-38 and instruction 39-45).
    pub fn allocate_for_inline(&self) -> VirtualRegisterGuard {
        // Skip reserved registers (32-38) and instruction registers (39-45)
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
                self.pending_clearing_inline
                    .lock()
                    .expect("Failed to lock virtual register allocator")
                    .push(i as u8 + RISCV_REGISTER_BASE);
                return VirtualRegisterGuard {
                    index: i as u8 + RISCV_REGISTER_BASE,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for inline: No registers left");
    }

    pub fn get_registers_for_reset(&self) -> Vec<u8> {
        // Assert that all inline registers (46+) have been dropped.
        // This function only returns inline registers from pending_clearing_inline,
        // so we only need to verify those are deallocated.
        // Skip reserved (32-38) and instruction (39-45) registers.
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

    // First allocatable register (after skipping reserved 32-38)
    const FIRST_ALLOC_REG: u8 = RISCV_REGISTER_BASE + NUM_RESERVED_VIRTUAL_REGISTERS as u8; // 39

    // First inline register (after reserved + instruction registers)
    const FIRST_INLINE_REG: u8 = RISCV_REGISTER_BASE
        + NUM_RESERVED_VIRTUAL_REGISTERS as u8
        + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8; // 46

    #[test]
    fn test_allocate_deallocate() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate();
            assert_eq!(*guard1, FIRST_ALLOC_REG); // register 39

            let guard2 = allocator.allocate();
            assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 40
        }

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG); // register 39 (reused)
    }

    #[test]
    fn test_deref() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_ALLOC_REG); // register 39
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

        // This should panic
        let _guard = allocator.allocate();
    }

    #[test]
    fn test_allocate_deallocate_inline() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate_for_inline();
            assert_eq!(*guard1, FIRST_INLINE_REG); // register 46

            let guard2 = allocator.allocate_for_inline();
            assert_eq!(*guard2, FIRST_INLINE_REG + 1); // register 47
        }

        let guard3 = allocator.allocate_for_inline();
        assert_eq!(*guard3, FIRST_INLINE_REG); // register 46 (reused)
    }

    #[test]
    fn test_deref_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate_for_inline();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_INLINE_REG); // register 46
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        // Inline registers start after reserved + instruction registers
        let num_inline_registers = NUM_VIRTUAL_REGISTERS
            - NUM_RESERVED_VIRTUAL_REGISTERS
            - NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        for i in 0..num_inline_registers {
            let guard = allocator.allocate_for_inline();
            assert_eq!(*guard, FIRST_INLINE_REG + i as u8);
            guards.push(guard);
        }

        // This should panic
        let _guard = allocator.allocate_for_inline();
    }

    #[test]
    fn test_combined_allocate_and_inline() {
        let allocator = VirtualRegisterAllocator::new();
        // Allocate some instruction registers (39-45)
        let guard1 = allocator.allocate();
        assert_eq!(*guard1, FIRST_ALLOC_REG); // register 39

        let guard2 = allocator.allocate();
        assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 40

        // Allocate some inline registers (46+)
        let inline_guard1 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard1, FIRST_INLINE_REG); // register 46

        let inline_guard2 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard2, FIRST_INLINE_REG + 1); // register 47

        // Allocate more instruction registers
        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG + 2); // register 41

        // Drop some guards and reallocate
        drop(guard2);
        drop(inline_guard1);

        // Should reuse the freed slots
        let guard4 = allocator.allocate();
        assert_eq!(*guard4, FIRST_ALLOC_REG + 1); // register 40 (reused)

        let inline_guard3 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard3, FIRST_INLINE_REG); // register 46 (reused)
    }

    #[test]
    fn test_csr_to_virtual_register() {
        let allocator = VirtualRegisterAllocator::new();

        // Test all supported CSRs
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

        // Test unsupported CSR
        assert_eq!(allocator.csr_to_virtual_register(0x999), None);
    }

    #[test]
    fn test_is_csr_ecall() {
        let allocator = VirtualRegisterAllocator::new();

        // Default is false
        assert!(!allocator.is_csr_ecall());

        // Set to true
        allocator.set_is_csr_ecall(true);
        assert!(allocator.is_csr_ecall());

        // Set back to false
        allocator.set_is_csr_ecall(false);
        assert!(!allocator.is_csr_ecall());
    }
}
