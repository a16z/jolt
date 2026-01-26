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
/// - Register 32: Reserved for LR/SC reservation address (persistent, never allocated)
/// - Registers 33-39: Temporary registers for inline sequences (allocate())
/// - Registers 40+: Registers for larger inlines (allocate_for_inline())
///
/// Register 32 is skipped by allocate() and allocate_for_inline() to ensure
/// it persists across instructions for LR/SC atomic operations.
const RESERVATION_REGISTER: u8 = RISCV_REGISTER_BASE; // register 32

/// Number of reserved virtual registers that are NOT allocated.
/// Currently just the reservation register (32).
const NUM_RESERVED_VIRTUAL_REGISTERS: usize = 1;

#[derive(Debug, Clone)]
pub struct VirtualRegisterAllocator {
    allocated: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
    /// At the end of the inline execution all registers have to be reset to 0
    /// This variable tracks which registers were allocated during inline execution
    /// when a register is allocated, this is set to true
    /// when a register is deallocated, this is not set to false (since it still needs to be reset)
    pending_clearing_inline: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
            pending_clearing_inline: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
        }
    }

    /// Get the reservation register (register 32) used for LR/SC operations.
    /// This register holds the memory address reserved by LR instructions
    /// and is checked by SC instructions.
    pub fn reservation_register(&self) -> u8 {
        RESERVATION_REGISTER
    }

    /// Allocate virtual register that can be used in the inline sequence of
    /// an instruction. Skips reserved registers (currently just register 32).
    pub(crate) fn allocate(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(NUM_RESERVED_VIRTUAL_REGISTERS) // Skip reservation register (32)
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
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
    /// Skips reserved registers (32) and instruction registers.
    pub fn allocate_for_inline(&self) -> VirtualRegisterGuard {
        // Skip reserved registers and instruction registers
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
                    .expect("Failed to lock virtual register allocator")[i] = true;
                return VirtualRegisterGuard {
                    index: i as u8 + RISCV_REGISTER_BASE,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for inline: No registers left");
    }

    pub fn get_registers_for_reset(&self) -> Vec<u8> {
        // Assert that all inline registers have been dropped.
        // Skip reserved registers and instruction registers.
        let skip_count = NUM_RESERVED_VIRTUAL_REGISTERS + NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        assert!(
            self.allocated
                .lock()
                .expect("Failed to lock virtual register allocator")
                .iter()
                .skip(skip_count)
                .all(|allocated| !*allocated),
            "All allocated virtual registers have to be dropped before inline finalization"
        );
        // Return the list of registers that need to be reset and clear the pending array
        let mut pending = self
            .pending_clearing_inline
            .lock()
            .expect("Failed to lock virtual register allocator");
        let result = pending
            .iter()
            .enumerate()
            .filter(|(_, p)| **p)
            .map(|(i, _)| i as u8 + RISCV_REGISTER_BASE)
            .collect::<Vec<u8>>();
        *pending = [false; NUM_VIRTUAL_REGISTERS];
        result
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

    // First allocatable register (after skipping reserved register 32)
    const FIRST_ALLOC_REG: u8 = RISCV_REGISTER_BASE + NUM_RESERVED_VIRTUAL_REGISTERS as u8; // 33

    // First inline register (after reserved + instruction registers)
    const FIRST_INLINE_REG: u8 = RISCV_REGISTER_BASE
        + NUM_RESERVED_VIRTUAL_REGISTERS as u8
        + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8;

    #[test]
    fn test_reservation_register() {
        let allocator = VirtualRegisterAllocator::new();
        assert_eq!(allocator.reservation_register(), RISCV_REGISTER_BASE); // register 32
    }

    #[test]
    fn test_allocate_deallocate() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate();
            assert_eq!(*guard1, FIRST_ALLOC_REG); // register 33

            let guard2 = allocator.allocate();
            assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 34
        }

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG); // register 33 (reused)
    }

    #[test]
    fn test_deref() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate();
        let index: u8 = *guard;
        assert_eq!(index, FIRST_ALLOC_REG); // register 33
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
            assert_eq!(*guard1, FIRST_INLINE_REG);

            let guard2 = allocator.allocate_for_inline();
            assert_eq!(*guard2, FIRST_INLINE_REG + 1);
        }

        let guard3 = allocator.allocate_for_inline();
        assert_eq!(*guard3, FIRST_INLINE_REG); // reused
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
        // Allocate some instruction registers (33+)
        let guard1 = allocator.allocate();
        assert_eq!(*guard1, FIRST_ALLOC_REG); // register 33

        let guard2 = allocator.allocate();
        assert_eq!(*guard2, FIRST_ALLOC_REG + 1); // register 34

        // Allocate some inline registers
        let inline_guard1 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard1, FIRST_INLINE_REG);

        let inline_guard2 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard2, FIRST_INLINE_REG + 1);

        // Allocate more instruction registers
        let guard3 = allocator.allocate();
        assert_eq!(*guard3, FIRST_ALLOC_REG + 2); // register 35

        // Drop some guards and reallocate
        drop(guard2);
        drop(inline_guard1);

        // Should reuse the freed slots
        let guard4 = allocator.allocate();
        assert_eq!(*guard4, FIRST_ALLOC_REG + 1); // register 34 (reused)

        let inline_guard3 = allocator.allocate_for_inline();
        assert_eq!(*inline_guard3, FIRST_INLINE_REG); // reused
    }
}
