use common::constants::{
    RISCV_REGISTER_COUNT, VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT,
};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize =
    VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT as usize;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

/// Layout of virtual registers:
/// - Register 32: Reservation address for LR/SC (persistent, not allocated)
/// - Register 33: Trap handler address (persistent, not allocated)
/// - Registers 34-40: Temporary registers for inline sequences (allocate())
/// - Registers 41+: Registers for larger inlines (allocate_for_inline())
///
/// The reserved registers (32, 33) are at the front but skipped by allocate()
/// to ensure they persist across instructions.

/// Register 32 is used for LR/SC reservation address.
const RESERVATION_REGISTER: u8 = RISCV_REGISTER_BASE; // register 32

/// Register 33 stores the trap handler address.
/// Set at boot via CSR ECALL, used by trap-taking ECALLs to verify target.
const TRAP_HANDLER_REGISTER: u8 = RISCV_REGISTER_BASE + 1; // register 33

/// Number of reserved virtual registers that are NOT allocated (reservation + trap handler).
/// allocate() skips these and starts from register 34.
const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 2;

#[derive(Debug, Clone)]
pub struct VirtualRegisterAllocator {
    allocated: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
    /// At the end of the inline execution all registers have to be reset to 0
    /// This variable tracks which registers were allocated during inline execution
    pending_clearing_inline: Arc<Mutex<Vec<u8>>>,
    /// Tracks whether the last ECALL took a trap (for EXIT syscalls).
    /// Set by ECALL.trace(), read by ECALL.inline_sequence() to return correct length.
    last_ecall_trap_taken: Arc<Mutex<bool>>,
    /// Tracks whether the current ECALL is a CSR ECALL (for trap handler setup).
    /// Set by cpu.handle_syscall(), read by ECALL.trace() to determine inline sequence.
    is_csr_ecall: Arc<Mutex<bool>>,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
            pending_clearing_inline: Arc::new(Mutex::new(Vec::new())),
            last_ecall_trap_taken: Arc::new(Mutex::new(false)),
            is_csr_ecall: Arc::new(Mutex::new(false)),
        }
    }

    /// Set whether the last ECALL took a trap (for EXIT syscalls).
    /// Called by ECALL.trace() to communicate with inline_sequence().
    pub fn set_last_ecall_trap_taken(&self, value: bool) {
        *self
            .last_ecall_trap_taken
            .lock()
            .expect("Failed to lock last_ecall_trap_taken") = value;
    }

    /// Get whether the last ECALL took a trap.
    /// Called by ECALL.inline_sequence() to determine return length.
    pub fn last_ecall_trap_taken(&self) -> bool {
        *self
            .last_ecall_trap_taken
            .lock()
            .expect("Failed to lock last_ecall_trap_taken")
    }

    /// Get the reservation register (register 32) used for LR/SC operations.
    /// This register holds the memory address reserved by LR instructions
    /// and is checked by SC instructions.
    pub fn reservation_register(&self) -> u8 {
        RESERVATION_REGISTER
    }

    /// Get the trap handler register (register 33) used for storing trap handler address.
    /// Set at boot via CSR ECALL, used by trap-taking ECALLs to verify target.
    pub fn trap_handler_register(&self) -> u8 {
        TRAP_HANDLER_REGISTER
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
    /// an instruction. Skips reserved registers (32, 33) and uses registers 34-40.
    pub(crate) fn allocate(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(NUM_RESERVED_VIRTUAL_REGISTERS) // Skip registers 32, 33
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS) // Take 7 registers (34-40)
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
    /// Uses registers 41+ (skips reserved 32-33 and instruction 34-40).
    pub fn allocate_for_inline(&self) -> VirtualRegisterGuard {
        // Skip reserved registers (32-33) and instruction registers (34-40)
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
        // Assert that all registers have been dropped
        assert!(
            self.allocated
                .lock()
                .expect("Failed to lock virtual register allocator")
                .iter()
                .skip(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
                .all(|allocated| !*allocated),
            "All allocated virtual registers have to be dropped before inline finalization"
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

    // First allocatable register (after skipping reserved 32, 33)
    const FIRST_ALLOC_REG: u8 = RISCV_REGISTER_BASE + NUM_RESERVED_VIRTUAL_REGISTERS as u8; // 34

    // First inline register (after reserved + instruction registers)
    const FIRST_INLINE_REG: u8 = RISCV_REGISTER_BASE
        + NUM_RESERVED_VIRTUAL_REGISTERS as u8
        + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8; // 41

    #[test]
    fn test_allocate_deallocate() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate();
            assert_eq!(*guard1, FIRST_ALLOC_REG); // register 34

            let guard2 = allocator.allocate();
            assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 35
        }

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG); // register 34 (reused)
    }

    #[test]
    fn test_deref() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_ALLOC_REG); // register 34
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
            assert_eq!(*guard1, FIRST_INLINE_REG); // register 41

            let guard2 = allocator.allocate_for_inline();
            assert_eq!(*guard2, FIRST_INLINE_REG + 1); // register 42
        }

        let guard3 = allocator.allocate_for_inline();
        assert_eq!(*guard3, FIRST_INLINE_REG); // register 41 (reused)
    }

    #[test]
    fn test_deref_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate_for_inline();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_INLINE_REG); // register 41
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        // Inline registers start after reserved + instruction registers
        let num_inline_registers =
            NUM_VIRTUAL_REGISTERS - NUM_RESERVED_VIRTUAL_REGISTERS - NUM_VIRTUAL_INSTRUCTION_REGISTERS;
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
        // Allocate some instruction registers (34-40)
        let guard1 = allocator.allocate();
        assert_eq!(*guard1, FIRST_ALLOC_REG); // register 34

        let guard2 = allocator.allocate();
        assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 35

        // Allocate some inline registers (41+)
        let inline_guard1 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard1, FIRST_INLINE_REG); // register 41

        let inline_guard2 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard2, FIRST_INLINE_REG + 1); // register 42

        // Allocate more instruction registers
        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG + 2); // register 36

        // Drop some guards and reallocate
        drop(guard2);
        drop(inline_guard1);

        // Should reuse the freed slots
        let guard4 = allocator.allocate();
        assert_eq!(*guard4, FIRST_ALLOC_REG + 1); // register 35 (reused)

        let inline_guard3 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard3, FIRST_INLINE_REG); // register 41 (reused)
    }
}
